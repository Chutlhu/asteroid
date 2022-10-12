import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import librosa as lb
from pathlib import Path

from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, PairwiseLogDetDiv, PairwiseNegSDR
from asteroid.data.wham_dataset import WhamDataset, normalize_tensor_wav
from asteroid.models import ConvTasNet
from asteroid.models.conv_tasnet import GPTasNet
from asteroid.utils import tensors_to_device
from asteroid.models import save_publishable

import matplotlib.pyplot as plt 
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)
parser.add_argument(
    "--test_dir", type=str, required=True, help="Test directory including the json files"
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=50, help="Number of audio examples to save, -1 means all"
)

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")    
    model = GPTasNet.from_pretrained(model_path, **conf['train_conf']['kernelnet'])
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = WhamDataset(
        conf["test_dir"],
        conf["task"],
        sample_rate=conf["sample_rate"],
        nondefault_nsrc=model.masker.n_src,
        segment=None,
    )  # Uses all segment length
    # Used to reorder sources only
    # loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    loss_func = PITLossWrapper(
        PairwiseLogDetDiv(
            inv_est=True, exp_dir=conf["exp_dir"],
            padding_to_remove=conf["filterbank"]["kernel_size"]),
        pit_from="pw_mtx")
    hlfpadd = loss_func.loss_func.half_padding_to_remove
    EPS = loss_func.loss_func.EPS

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf["exp_dir"], "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    
    # set for evaluation
    loss_func.training = False
    
    for idx in tqdm(range(len(test_set))):
        
        # get utt from wham data loader
        # mix is L
        # src is K x L
        mix, sources = tensors_to_device(test_set[idx], device=model_device)
        print('Mix shape', mix.shape)
        
        frame_length = int(conf['train_conf']['data']['segment_duration'] * conf['train_conf']['data']['sample_rate'])
        hop_length = frame_length//2
        
        ''' Pad the sources to avoit window artifacts '''
        pad = frame_length//2
        L = mix.shape[0]
        padded_mix = torch.concat([torch.zeros(pad).to(model_device), mix, torch.zeros(4*pad).to(model_device)])
        K, L = sources.shape
        
        padded_src = torch.concat([torch.zeros(K,pad).to(model_device), sources, torch.zeros(K, 4*pad).to(model_device)], dim=-1    )
        
        ''' Framing the mixture '''
        _mix = padded_mix.detach().cpu().numpy()
        mix_frames = lb.util.frame(_mix, frame_length=frame_length, hop_length=hop_length).T # N frames x Len frames
        N, T = mix_frames.shape
        mix_frames = torch.Tensor(mix_frames.copy()).to(model_device)
        print('Mix Frame shape:', mix_frames.shape)
        
        ''' Framing the sources '''
        print('Src shape', sources.shape)
        _src = padded_src.detach().cpu().numpy()
        src_frames = lb.util.frame(_src, frame_length=frame_length, hop_length=hop_length).T # N frames x Len frames x K
        src_frames = torch.Tensor(src_frames.copy()).to(model_device).permute(0,2,1) # NxTxK -> NxKxT
        print('Src Frame shape:', src_frames.shape) # Nframes x Ksources x T
        
        # check dimension
        try:
            assert mix_frames.shape == src_frames[:,0,:].shape
        except:
            print(mix_frames.shape)
            print(src_frames.shape)
            raise ValueError('Src and Mix different size')
        
        est_src_frames = torch.zeros_like(src_frames)
        
        ''' Estimate frames and reconstruct the wav file '''
        eye = None
        B = min(50, N) # arg_dic['train_conf']['training']['batch_size']
        for i in range(0, N, B):
            
            _mixture = mix_frames[i:i+B,:].clone()
            _targets = src_frames[i:i+B,:,:].clone()
            
            # Normalize the mixture only
            # the targets will preserve the original std
            m_std = _mixture.std(-1, keepdim=True)
            _mixture = normalize_tensor_wav(_mixture, eps=EPS, std=m_std)
            
            # Forward the network on the mixture.
            _est_cov, _, _ = model(_mixture)
            
            # Compute the filters
            _est_filters = torch.linalg.solve(
                    _est_cov.sum(dim=1, keepdim=True).transpose(-2, -1),
                    _est_cov.transpose(-2, -1)
                ).transpose(-2, -1)
            
            # Separation
            __mixture = _mixture[..., None, hlfpadd:-hlfpadd]
            _est_sources_frames = torch.einsum('...ij,...j->...i', _est_filters, __mixture)
            
            # Resolve permutation
            _compute_sdr = PairwiseNegSDR("sisdr")
            _negsdrs = _compute_sdr(_est_sources_frames, _targets[..., hlfpadd:-hlfpadd])
            _pitlosswrapper = PITLossWrapper(_compute_sdr)
            _min_loss, _batch_indices = _pitlosswrapper.find_best_perm(_negsdrs)
            est_src_frames[i:i+B,:,hlfpadd:-hlfpadd] = _pitlosswrapper.reorder_source(_est_sources_frames, _batch_indices)       
        
        # windowing
        win = torch.hann_window(frame_length, periodic=True, device=model_device)
        est_src_frames = est_src_frames * win[None,None,:]
        
        est_sources = torch.zeros_like(padded_src) # Ksourcs x Nsamples
        
        for n in range(N):
            # print(hop*n, hop*n+T)
            est_sources[:,hop_length*n:hop_length*n+T] += est_src_frames[n,:,:]
        
        # if idx // 100 == 0:
        #     plt.figure(figsize=(12,6))
        #     plt.subplot(211)
        #     plt.title(f"Sample{idx} - source:0")
        #     plt.plot(_sources[0,:], label='True')
        #     plt.plot(_est_sources[0,:], label='Est')
        #     plt.subplot(212)
        #     plt.title(f"Sample{idx} - source:1")
        #     plt.plot(_sources[1,:], label='True')
        #     plt.plot(_est_sources[1,:], label='Est')
        #     plt.savefig(os.path.join(conf["exp_dir"], "dummy_reconstruction.png"))
        #     plt.close()
        
        # path_to_wav = Path(conf['exp_dir']) / Path('test_wav')
        # path_to_wav.mkdir(parents=True,exist_ok=True)
        
        # sf.write(str(path_to_wav / Path(f"sample_{idx}_est.wav")), _est_sources.T, conf['sample_rate'])
        # sf.write(str(path_to_wav / Path(f"sample_{idx}_tgt.wav")), _sources.T, conf['sample_rate'])
        # sf.write(str(path_to_wav / Path(f"sample_{idx}_mix.wav")), _mix, conf['sample_rate'])
                
        est_sources_np = est_sources.cpu().data.numpy()      # KxN
        L = est_sources.shape[-1]
        sources_np = padded_src.cpu().data.numpy()[:,:L]     # KxN
        mix_np = padded_mix[None,:].cpu().data.numpy()[:,:L] # KxN
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=compute_metrics,
        )
        utt_metrics["mix_path"] = test_set.mix[idx][0]
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np[0], conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx + 1), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np):
                est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx + 1),
                    est_src,
                    conf["sample_rate"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(conf["exp_dir"], "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()
    print("Overall metrics :")
    pprint(final_results)
    with open(os.path.join(conf["exp_dir"], "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    model_dict = torch.load(model_path, map_location="cpu")
    os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
    publishable = save_publishable(
        os.path.join(conf["exp_dir"], "publish_dir"),
        model_dict,
        metrics=final_results,
        train_conf=train_conf,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf
    arg_dic["filterbank"] = train_conf["filterbank"]

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )
    
    print('ATTENTION: TESTING ON VALIDATION SET')
    arg_dic["test_dir"] = arg_dic['train_conf']['data']['valid_dir']
    print(arg_dic["test_dir"])
    main(arg_dic)

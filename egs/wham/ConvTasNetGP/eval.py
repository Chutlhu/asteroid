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
    "--model", type=str, required=True, help="Select best or last"
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=50, help="Number of audio examples to save, -1 means all"
)
parser.add_argument(
    "--batch_size", type=int, default=10, help="Batch size"
)

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf):
    
    suffix = conf['model']
    if conf['model'] == 'best':
        model_path = os.path.join(conf["exp_dir"], "best_model.pth")    
    elif conf['model'] == 'last':
        model_path = os.path.join(conf["exp_dir"], "last_model.pth")    
    else:
        raise ValueError('Model arg must be either "best" or "last"')
    print('Loading model path', model_path)
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
   
    pad_to_remove = model.half_pad_to_remove*2 + conf["filterbank"]["kernel_size"]
    half_pad_to_remove = pad_to_remove // 2
    loss_func = PITLossWrapper(
        PairwiseLogDetDiv(
            inv_est=True, exp_dir=conf["exp_dir"],
            padding_to_remove=pad_to_remove),
        pit_from="pw_mtx")
    EPS = loss_func.loss_func.EPS
    loss_func.loss_func.testing = True
    loss_func.loss_func.training = False
    model.loss = loss_func.loss_func

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf["exp_dir"], 'eval', f"{suffix}_examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    
    
    half_pad_to_remove = 395 #750
    
    for idx in tqdm(range(len(test_set))):
        
        # get utt from wham data loader
        # mix is L
        # src is K x L
        mix, sources = tensors_to_device(test_set[idx], device=model_device)
        
        frame_length = int(conf['train_conf']['data']['segment_duration'] * conf['train_conf']['data']['sample_rate'])
         
        ''' Pad the sources to avoit window artifacts '''
        L = mix.shape[0]
        padded_mix = torch.concat([torch.zeros(frame_length).to(model_device), mix, torch.zeros(frame_length).to(model_device)])
        K, L = sources.shape
        
        
        ''' Framing the mixture '''
        _mix = padded_mix.detach().cpu().numpy()
        mix_frames = lb.util.frame(_mix, 
                                   frame_length=frame_length, 
                                   hop_length=frame_length-2*half_pad_to_remove).T # N frames x Len frames
        Nx, Tx = mix_frames.shape
        mix_frames = torch.Tensor(mix_frames.copy()).to(model_device)
        # print('Mix Frame shape', mix_frames.shape)
        
        ''' Framing the sources '''
        # pad sources
        padded_src = torch.concat([torch.zeros(K,frame_length-half_pad_to_remove).to(model_device), sources, torch.zeros(K,frame_length).to(model_device)], dim=-1)
        _src = padded_src.detach().cpu().numpy()
        # frame sources
        src_frames = lb.util.frame(_src, 
                                   frame_length=frame_length-2*half_pad_to_remove, 
                                   hop_length=frame_length-2*half_pad_to_remove).T # N frames x Len frames x K
        src_frames = torch.Tensor(src_frames.copy()).to(model_device).permute(0,2,1) # NxTxK -> NxKxT
        # print('Src Frame shape:', src_frames.shape) # Nframes x Ksources x T

        Ns, K, Ts = src_frames.shape
        N = min(Ns,Nx)
        src_frames = src_frames[:N,...]
        mix_frames = mix_frames[:N,...]
        
        # # reconstruct to check
        __src = torch.concat([src_frames[n,...].sum(0) for n in range(N)], dim=0)
        __mix = torch.concat([mix_frames[n,half_pad_to_remove:-half_pad_to_remove] for n in range(N)], dim=0)
        
        # print(__src.shape)
        # print(__mix.shape)
        L = min(__src.shape[0], __mix.shape[0])
        assert torch.allclose(__src[:L], __mix[:L])
        
        # plt.figure(figsize=(12,6))
        # plt.plot(__src[:L], label='src')
        # plt.plot(__mix[:L], alpha=0.5, label='mix')
        # plt.legend()
        # plt.savefig('dummy_rec.png')
        
        # 1/0
        
        # plt.figure(figsize=(16,4))
        # off = 67
        # for i in range(10):
        #     plt.subplot(2,5,i+1)
        #     plt.plot(np.concatenate((np.zeros(half_pad_to_remove), src_frames[i+off,:,:].sum(0), np.zeros(half_pad_to_remove))))
        #     plt.plot(mix_frames[i+off,:], alpha=0.5)
        #     plt.xlim([750,1500])
        # plt.savefig('dummy_rec.png')
        
        est_src_frames = torch.zeros_like(src_frames)
        
        ''' Estimate frames and reconstruct the wav file '''
        eye = None
        B = min(conf['batch_size'], Nx) # arg_dic['train_conf']['training']['batch_size']
        for i in range(0, Nx, B):
            
            _mixture = mix_frames[i:i+B,:].clone()
            _targets = src_frames[i:i+B,:,:].clone()
            
            # Normalize the mixture only
            # the targets will preserve the original std
            m_std = _mixture.std(-1, keepdim=True)
            __mixture = normalize_tensor_wav(_mixture, eps=EPS, std=m_std)
            
            # Forward the network on the mixture.
            _est_cov, _, extra2 = model(__mixture)
            
            # print(model.half_pad_to_remove)
            # print(half_pad_to_remove)
            # print(_est_cov.shape)
            # print(__mixture.shape)
            # 1/0
            
            target_dim = __mixture.shape[-1] - 2*half_pad_to_remove
            P = (_est_cov.shape[-1] - target_dim) // 2
            __est_cov = _est_cov[:,:,P:-P,P:-P]
            
            # if idx == 0:
            #     print()
            #     print(P)
            #     print(__mixture[..., None, half_pad_to_remove:-half_pad_to_remove].shape)
                
            
            # Compute the filters
            _est_filters = torch.linalg.solve(
                    __est_cov.sum(dim=1, keepdim=True).transpose(-2, -1),
                    __est_cov.transpose(-2, -1)
                ).transpose(-2, -1)
            
            # Separation
            _est_sources = torch.einsum('...ij,...j->...i', _est_filters, __mixture[..., None, half_pad_to_remove:-half_pad_to_remove])
            _est_sources *= m_std[:,:,None]
            
            # print(_est_sources.shape)
            # print(_targets.shape)
            
            # Resolve permutation
            _compute_sdr = PairwiseNegSDR("sisdr")
            _negsdrs = _compute_sdr(_est_sources, _targets)
            _pitlosswrapper = PITLossWrapper(_compute_sdr)
            _min_loss, _batch_indices = _pitlosswrapper.find_best_perm(_negsdrs)
            est_src_frames[i:i+B,:,:] = _pitlosswrapper.reorder_source(_est_sources, _batch_indices)       
        
        # windowing
        # win = torch.hann_window(frame_length, periodic=True, device=model_device)
        # est_src_frames = est_src_frames * win[None,None,:]
        est_sources = torch.concat([est_src_frames[f,:,:] for f in range(est_src_frames.shape[0])],dim=-1)
        # sources = torch.concat([src_frames[f,:,:] for f in range(src_frames.shape[0])],dim=-1)
        # mix = torch.concat([mix_frames[f,:] for f in range(mix_frames.shape[0])],dim=-1)
        
        
        # plt.figure(figsize=(12,6))
        # plt.subplot(211)
        # plt.title(f"Sample{idx} - source:0")
        # plt.plot(sources[0,:].cpu(), label='True')
        # plt.plot(est_sources[0,:].cpu(), label='Est')
        # plt.subplot(212)
        # plt.title(f"Sample{idx} - source:1")
        # plt.plot(sources[1,:].cpu(), label='True')
        # plt.plot(est_sources[1,:].cpu(), label='Est')
        # plt.savefig(os.path.join(conf["exp_dir"], f"{suffix}_dummy_reconstruction.png"))
        # plt.close()
        # 1/0
        
        # path_to_wav = Path(conf['exp_dir']) / Path('test_wav')
        # path_to_wav.mkdir(parents=True,exist_ok=True)
        
        # sf.write(str(path_to_wav / Path(f"sample_{idx}_est.wav")), _est_sources.T, conf['sample_rate'])
        # sf.write(str(path_to_wav / Path(f"sample_{idx}_tgt.wav")), _sources.T, conf['sample_rate'])
        # sf.write(str(path_to_wav / Path(f"sample_{idx}_mix.wav")), _mix, conf['sample_rate'])
                
        est_sources_np = est_sources.cpu().data.numpy()      # KxN
        sources_np = _src             # KxN
        mix_np = _mix[None,half_pad_to_remove:]       # KxN
        L = min(est_sources_np.shape[-1], sources_np.shape[-1], mix_np.shape[-1])
        
        est_sources_np = est_sources_np[:,:L]
        sources_np = sources_np[:,:L]
        mix_np = mix_np[:,:L]
        
        
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=compute_metrics,
            average=True,
        )
        # n_srcs = sources_np.shape[0]
        # for i in range(n_srcs):
            
        utt_metrics["mix_path"] = test_set.mix[idx][0]
        series_list.append(pd.Series(utt_metrics))
        

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            print(conf['sample_rate'])
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
    all_metrics_df.to_csv(os.path.join(conf["exp_dir"], 'eval', f"{suffix}_all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()
    print("Overall metrics :")
    pprint(final_results)
    with open(os.path.join(conf["exp_dir"], 'eval', f"{suffix}_final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    model_dict = torch.load(model_path, map_location="cpu")
    os.makedirs(os.path.join(conf["exp_dir"], 'eval', f"{suffix}_publish_dir"), exist_ok=True)
    publishable = save_publishable(
        os.path.join(conf["exp_dir"], 'eval', f"{suffix}_publish_dir"),
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
    
    # print('ATTENTION: TESTING ON VALIDATION SET')
    # arg_dic["test_dir"] = arg_dic['train_conf']['data']['valid_dir']
    print(arg_dic["test_dir"])
    main(arg_dic)

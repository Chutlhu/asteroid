from turtle import forward
import torch
import torch.nn as nn
from asteroid_filterbanks import make_enc_dec
from ..masknn import TDConvNet
from asteroid.masknn.convolutional import GPTDConvNet
from .base_models import BaseEncoderMaskerDecoder, jitable_shape, _shape_reconstructed, pad_x_to_y, _unsqueeze_to_3d
import warnings

import matplotlib.pyplot as plt

class ConvTasNet(BaseEncoderMaskerDecoder):
    """ConvTasNet separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        causal (bool, optional) : Whether or not the convolutions are causal.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
          for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
          https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="sigmoid",
        in_chan=None,
        causal=False,
        fb_name="free",
        kernel_size=16,
        n_filters=512,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        if causal and norm_type not in ["cgLN", "cLN"]:
            norm_type = "cLN"
            warnings.warn(
                "In causal configuration cumulative layer normalization (cgLN)"
                "or channel-wise layer normalization (chanLN)  "
                f"must be used. Changing {norm_type} to cLN"
            )
        # Update in_chan
        masker = GPTDConvNet(
            n_feats,
            n_src,
            out_chan=out_chan,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            norm_type=norm_type,
            mask_act=mask_act,
            causal=causal,
        )

        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)


class VADNet(ConvTasNet):
    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.sigmoid(self.decoder(masked_tf_rep))
    
def LinearTanh(n_in, n_out):
    block = nn.Sequential(
      nn.Linear(n_in, n_out),
      nn.Tanh()
    )
    return block
    
def LinearPReLu(n_in, n_out):
    block = nn.Sequential(
      nn.Linear(n_in, n_out),
      nn.PReLU()
    )
    return block
    
class DeepKernel(nn.Module):
    
    def __init__(self, dim_layers=[]):
        super().__init__()
        
        num_layers = len(dim_layers)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearPReLu(dim_layers[l], dim_layers[l+1]))
            
        blocks.append(
            nn.Sequential(
                nn.Linear(dim_layers[-2], dim_layers[-1]),
                nn.PReLU()
            )
        )
        
        self.mlp = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.mlp(x)


class GPTasNet(ConvTasNet):
    
    def __init__(self, n_src, out_chan=None, n_blocks=8, n_repeats=3, bn_chan=128, hid_chan=512, skip_chan=128, conv_kernel_size=3, norm_type="gLN", mask_act="sigmoid", in_chan=None, causal=False, fb_name="free", kernel_size=16, n_filters=512, stride=8, encoder_activation=None, sample_rate=8000, **fb_kwargs):
        super().__init__(n_src, out_chan, n_blocks, n_repeats, bn_chan, hid_chan, skip_chan, conv_kernel_size, norm_type, mask_act, in_chan, causal, fb_name, kernel_size, n_filters, stride, encoder_activation, sample_rate, **fb_kwargs)
        
        self.deep_kernel = DeepKernel([256] + [128] + [64])
        self.weight_c1 = torch.nn.Parameter(0.25*torch.ones(1))
        self.weight_c2 = torch.nn.Parameter(0.25*torch.ones(1))
    
    def forward(self, wav):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        

        # print()
        # print(wav.shape)
        # print('TF repr:', tf_rep.shape)
        # print('Mask   :', est_masks.shape)
        # 1/0
    
        # Original ConvTasNet
        # est_masks = self.forward_masker(tf_rep)
        # masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        # decoded = self.forward_decoder(masked_tf_rep)
        # reconstructed = pad_x_to_y(decoded, wav)
        # return _shape_reconstructed(reconstructed, shape)
        
        # Feed the mask to the kernel function:
        z1, z2 = est_masks.split(1, dim=1)
        
        phi_z1 = self.deep_kernel(z1.permute(0,1,3,2)) # BxCxFxT -> BxCxTxF
        phi_z2 = self.deep_kernel(z2.permute(0,1,3,2)) # BxCxFxT -> BxCxTxF
        # phi_z1 = z1.permute(0,1,3,2)
        # phi_z2 = z2.permute(0,1,3,2)
        
        # print(phi_z1)
        # print(phi_z2)
        
        # C1 = torch.diag_embed(torch.sum(phi_z1**2, dim=-1))
        # C2 = torch.diag_embed(torch.sum(phi_z2**2, dim=-1))
        
        
        # C1exp = torch.norm(phi_z1[..., None, :, :] - phi_z1[..., :, None, :], dim=-1)**2
        # C2exp = torch.norm(phi_z2[..., None, :, :] - phi_z2[..., :, None, :], dim=-1)**2
        # print(C1exp.shape)
        
        C1 = torch.einsum('...if,...jf->...ij',phi_z1,phi_z1)
        C2 = torch.einsum('...if,...jf->...ij',phi_z2,phi_z2)
        # C1 = C1 + self.weight_c1*torch.exp(C1exp)
        # C2 = C2 + self.weight_c2*torch.exp(C2exp)
        
        cov_est_targets = torch.concat([C1, C2], dim=1)
        
        # inv_sum_cov_est_targets = torch.linalg.inv(cov_est_targets.sum(dim=1, keepdim=True))
        # filters = torch.einsum('...ij,...jk->...ik', cov_est_targets, inv_sum_cov_est_targets)
        
        # padding_to_remove = 33 // 2 
        # _wav = wav[..., padding_to_remove:-padding_to_remove]
        # estimates = torch.einsum('...ij,...j->...i', filters, _wav)
        
        # estimates = torch.concat((torch.zeros_like(estimates[:,:,:padding_to_remove]), 
        #                           estimates, 
        #                           torch.zeros_like(estimates[:,:,-padding_to_remove:])), dim=-1)
        
        return (cov_est_targets, ((z1, z2), (phi_z1, phi_z2)))
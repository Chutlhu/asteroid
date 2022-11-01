import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import make_enc_dec
from ..masknn import TDConvNet
from asteroid.masknn.convolutional import GPTDConvNet
from .base_models import BaseEncoderMaskerDecoder, jitable_shape, _shape_reconstructed, pad_x_to_y, _unsqueeze_to_3d
import warnings
import numpy as np

import time
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

def LinearELU(n_in, n_out):
    block = nn.Sequential(
      nn.Linear(n_in, n_out),
      nn.ELU()
    )
    return block

class DeepSpectralKernel(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=64):
        super(DeepSpectralKernel, self).__init__()

        self.net = nn.Sequential(
                    nn.Sequential(nn.Linear(in_dim, out_dim), nn.Linear(in_dim, out_dim)),
                    # nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)),
                    # nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Linear(hidden_dim, out_dim))
        )

        for layers in self.net:
            for layer in layers:
                torch.nn.init.normal_(layer.weight, 0 , 1)
                torch.nn.init.uniform_(layer.bias, 0 , 2*np.pi)

    def forward(self, x):
        for layer in self.net:
            # a = 1 / np.sqrt(2 * layer[0].weight.shape[-1])
            x = (torch.cos(layer[0](x * 2 * np.pi)) + torch.cos(layer[1](x * 2 * np.pi))) / 2
        return x

class DeepKernel(nn.Module):

    def __init__(self, dim_layers=[], act='none'):
        super().__init__()

        num_layers = len(dim_layers)

        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearELU(dim_layers[l], dim_layers[l+1]))

        blocks.append(
            nn.Sequential(
                nn.Linear(dim_layers[-2], dim_layers[-1], bias=False),
            )
        )
        
        if act == 'tanh':
            blocks[-1].append(nn.Tanh())
        elif act == 'none':
            pass
        elif act == 'prelu':
            blocks[-1].append(nn.PReLU())
        elif act == 'elu':
            blocks[-1].append(nn.ELU())
        elif act == 'sigmoid':
            blocks[-1].append(nn.Sigmoid())
        elif act == 'softplus':
            blocks[-1].append(nn.SoftPlus())
        else:
            raise NotImplementedError

        self.mlp = nn.Sequential(*blocks)
        
        # for layers in self.mlp:
        #     # torch.nn.init.normal_(layer.weight, 0 , 1)
        #     linear = layers[0]
        #     if linear.bias is not None:
                # torch.nn.init.constant_(linear.bias, 0)


    def forward(self, x):
        return self.mlp(x)



def square_difference(est_latents):
    """
    Args:
        est_latents (torch.Tensor): Estimated latent vectors
            (batch, n_src, feat, seq)
    """
    _inner = torch.einsum('...ji,...jk->...ik', est_latents, est_latents)
    _sq_vec = (est_latents ** 2).sum(dim=-2, keepdim=True)
    _sq_norms = _sq_vec + _sq_vec.transpose(-2, -1) - 2 * _inner
    # _sq_norms = torch.clamp(_sq_norms, min=1e-6)
    return _sq_norms

class GPTasNet(ConvTasNet):

    def __init__(self, k_n_layers, k_hid_size, k_out_size, n_src, out_chan=None, n_blocks=8, n_repeats=3, bn_chan=128, hid_chan=512, skip_chan=128, conv_kernel_size=3, norm_type="gLN", mask_act="sigmoid", in_chan=None, causal=False, fb_name="free", kernel_size=16, n_filters=512, stride=8, encoder_activation=None, sample_rate=8000, **fb_kwargs):
        super().__init__(n_src, out_chan, n_blocks, n_repeats, bn_chan, hid_chan, skip_chan, conv_kernel_size, norm_type, mask_act, in_chan, causal, fb_name, kernel_size, n_filters, stride, encoder_activation, sample_rate, **fb_kwargs)

        self.mode = 'vae'
        if self.mode == 'krl':
            self.deep_kernel = DeepKernel([256] + k_n_layers * [k_hid_size] + [k_out_size])
        if self.mode == 'vae':
            # self.backbone = DeepKernel([256] + [k_hid_size], act='elu')
            self.vae_mean = DeepKernel([256] + k_n_layers * [k_hid_size] + [k_out_size], act='tanh')
            self.vae_lvar = DeepKernel([256] + k_n_layers * [k_hid_size] + [k_out_size], act='none')
            # self.deep_kernel = DeepKernel([k_out_size] + [k_out_size] + [k_out_size], act='none')
        self.k_out_size = k_out_size

        self.halfpadd = None
        self.w_diag = torch.nn.Parameter(0.001*torch.ones(n_src))
        self.eye = None
        self.EPS = 1e-6
        self.trainer = None
        
        self.receptive_field =  ((n_repeats*(conv_kernel_size - 1)*( 2**n_blocks-1))*stride + kernel_size) // 2
        self.half_pad_to_remove = self.receptive_field // 2
        
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(mean).to(var.device)          # sampling epsilon        
        z = mean + torch.mean(var, dim=-1, keepdim=True)*epsilon # reparameterization trick
        return z


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

        # if self.loss.training and self.trainer.current_epoch < 0:
        #     self.encoder.requires_grad_(False)
        #     self.masker.requires_grad_(False)
        # else:
        #     self.encoder.requires_grad_(True)
        #     self.masker.requires_grad_(True)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)

        # z = est_masks.split(1, dim=1)
        z = est_masks[...,self.half_pad_to_remove:-self.half_pad_to_remove].permute(0,1,3,2)  # BxCxFxT -> BxCxTxF
        # z2 = z2[...,self.half_pad_to_remove:-self.half_pad_to_remove].permute(0,1,3,2)  # BxCxFxT -> BxCxTxF
        
        B, K, T, F = z.shape
        if self.eye is None:
            self.eye = torch.eye(T,T)[None,None,...].to(wav.device)
        
        # RepamTrick
        if self.mode == 'vae':
            # _z = self.backbone(z)
            means = self.vae_mean(z) # BxCxTxH
            lvars = 10*self.vae_lvar(z) # BxCxTxH
            # import ipdb; ipdb.set_trace()
            phi_z = self.reparameterization(means, torch.exp(0.5 * lvars)) # takes exponential function (log var -> var)
            # phi_z = self.deep_kernel(phi_z)
            # import ipdb; ipdb.set_trace()
            
        elif self.mode == 'krl':
            # Feed the mask to the kernel function:
            phi_z = self.deep_kernel(z) # BxCxTxF
            
        
        cov_est_targets = torch.einsum('...if,...jf->...ij', phi_z, phi_z) + self.w_diag[None,:,None,None] * self.eye #+ self.const * self.full
        # C2b = torch.einsum('...if,...jf->...ij', phi_z2, phi_z2) + self.w_diag * self.eye #+ self.const * self.full
        
        # Cx = self.eye * self.EPS + torch.einsum('...i,...j->...ij', wav[...,self.halfpadd:-self.halfpadd], wav[...,self.halfpadd:-self.halfpadd])
        # C1a += Cx
        # C2b += Cx
        

        # assert (C1a.shape[-2], C1a.shape[-1])  == (z1.shape[-2], z1.shape[-2])
        # assert C1a.shape == C2b.shape
        

        # print(C1exp.shape)
        # |x - y|^2 = (x - y)^T (x - y) = x^T x + y^T y - 2 x^T y
        # z1, z2 = est_masks.split(1, dim=1)
        # phi_z1 = self.deep_kernel2(z1.permute(0,1,3,2)) # BxCxFxT -> BxCxTxF
        # phi_z2 = self.deep_kernel2(z2.permute(0,1,3,2)) # BxCxFxT -> BxCxTxF
        # C1exp = square_difference(phi_z1.permute(0,1,3,2)).permute(0,1,3,2) # BxCxTxF -> BxCxFxT -> BxCxTxF
        # C2exp = square_difference(phi_z2.permute(0,1,3,2)).permute(0,1,3,2) # BxCxTxF -> BxCxFxT -> BxCxTxF

        # C1 = C1a + self.weight_c1*torch.exp(-C1exp)
        # C2 = C2b + self.weight_c1*torch.exp(-C2exp)

        
        # cov_est_targets = torch.concat([C1a, C2b], dim=1)
        
        extra = {
            'w_diag' : self.w_diag.detach().cpu().numpy(),
            'R'      : self.half_pad_to_remove
        }
        if self.mode == 'vae':
            extra['means'] = means
            extra['lvars'] = lvars

        return (cov_est_targets, (z, phi_z), extra)
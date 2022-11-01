from lib2to3.pytree import convert
import os
import time
import soundfile as sf
import numpy as np
import scipy as sp
import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt
from asteroid.losses import PITLossWrapper
from asteroid.losses.sdr import PairwiseNegSDR
from asteroid.losses.mvar_gauss import NLL_Multivariate_Gaussian


class PairwiseLogDetDiv(_Loss):
    r"""
    Shape:
        - est_targets : :math:`(batch, nsrc, segment_len, segment_len)`.
        - targets: :math:`(batch, nsrc, segment_len)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`. Pairwise losses.
    """
    def __init__(self, padding_to_remove, extra_warmup_loss_diag=0, inv_est=True, EPS=1e-6, exp_dir=None):
        super().__init__()
        self.EPS = EPS
        self.inv_est = inv_est
        self.half_padding_to_remove = padding_to_remove // 2
        self.identity_mat = None
        self.check_pairwise_loss_indexing = False
        self.exp_dir = exp_dir
        self.trainer = None
        self.extra_warmup_loss_diag = extra_warmup_loss_diag
        self.testing = False
        self.nll = NLL_Multivariate_Gaussian(padding_to_remove, mode="vanilla", simplified=False,
                inv_est=True, EPS=EPS, exp_dir=exp_dir, use_cupy=False)
        self.loss_terms = []

    def compute_log_det_div_mix(self, cov_est_targets, cov_targets, cov_mixture):
        # src_log_det_div = self.compute_log_det_div(cov_est_targets, cov_targets)
        _sum_cov_est_targets = cov_est_targets.sum(dim=1, keepdim=True)
        _cov = torch.linalg.solve(
            _sum_cov_est_targets.transpose(-2, -1),
            cov_mixture.transpose(-2, -1)).transpose(-2, -1)
        _tr = torch.diagonal(_cov, dim1=-2, dim2=-1).sum(dim=-1)
        _sign, _logdet = torch.linalg.slogdet(_cov)
        log_det_div = _tr - _logdet - cov_est_targets.size(-1)
        return log_det_div[..., None] / cov_est_targets.size(1)


    def compute_log_det_div(self, cov_est_targets, cov_targets):
        assert cov_est_targets.shape == cov_targets.shape
        _cov_targets = torch.unsqueeze(cov_targets, dim=1)
        _cov_est_targets = torch.unsqueeze(cov_est_targets, dim=2)
        if self.inv_est:
            # _inv_cov_est_targets = torch.linalg.solve(_cov_est_targets, self.identity_mat)
            # _inv_cov_est_targets = torch.linalg.inv(_cov_est_targets)
            # _cov = torch.einsum('...ij,...jk->...ik', _cov_targets, _inv_cov_est_targets)
            _cov = torch.linalg.solve(_cov_est_targets.transpose(-2, -1), _cov_targets.transpose(-2, -1)).transpose(-2, -1)
        else:
            # _inv_cov_targets = torch.linalg.solve(_cov_targets, self.identity_mat)
            # _inv_cov_targets = torch.linalg.inv(_cov_targets)
            # _cov = torch.einsum('...ij,...jk->...ik', _cov_est_targets, _inv_cov_targets)
            _cov = torch.linalg.solve(_cov_targets.transpose(-2, -1), _cov_est_targets.transpose(-2, -1)).transpose(-2, -1)
        _tr = torch.diagonal(_cov, dim1=-2, dim2=-1).sum(dim=-1)
        _sign, _logdet = torch.linalg.slogdet(_cov)
        log_det_div = _tr - _logdet - cov_est_targets.size(-1)
        if not self.check_pairwise_loss_indexing and self.inv_est:
            ''' check based on the code of get_pw_losses(...)
                pair_wise_losses[:, est_idx, target_idx] = loss_func(est_src, target_src, **kwargs) '''
            # __cov = torch.einsum('...ij,...jk->...ik', cov_targets[:, 1], torch.linalg.inv(cov_est_targets[:, 0]))
            __cov = torch.linalg.solve(cov_est_targets[:, 0].transpose(-2, -1), cov_targets[:, 1].transpose(-2, -1)).transpose(-2, -1)
            __tr = torch.diagonal(__cov, dim1=-2, dim2=-1).sum(dim=-1)
            _, __logdet = torch.linalg.slogdet(__cov)
            _log_det_div = __tr - __logdet - cov_est_targets.size(-1)
            assert torch.allclose(log_det_div[:, 0, 1].detach(), _log_det_div.detach(), rtol=1e-1)
            self.check_pairwise_loss_indexing = True
        return log_det_div
     
        
    def compute_itakura_saito_diag(self, cov_est_targets, cov_targets):
        _diag_cov_targets = torch.unsqueeze(
            torch.diagonal(cov_targets, dim1=-2, dim2=-1), dim=1)
        _diag_cov_est_targets = torch.unsqueeze(
            torch.diagonal(cov_est_targets, dim1=-2, dim2=-1), dim=2)
        if self.inv_est:
            _tmp = _diag_cov_targets / _diag_cov_est_targets
            log_det_div = _tmp.sum(dim=-1) \
                - _diag_cov_targets.log().sum(dim=-1) \
                + _diag_cov_est_targets.log().sum(dim=-1) \
                - cov_est_targets.size(-1)
        else:
            _tmp = _diag_cov_est_targets / _diag_cov_targets
            log_det_div = _tmp.sum(dim=-1) \
                + _diag_cov_targets.log().sum(dim=-1) \
                - _diag_cov_est_targets.log().sum(dim=-1) \
                - cov_est_targets.size(-1)
        return log_det_div

    def compute_log_det_div_p(self, cov_est_targets, cov_targets, weight_ld=1.0):
        _cov_targets = torch.unsqueeze(cov_targets, dim=1)
        _cov_est_targets = torch.unsqueeze(cov_est_targets, dim=2)
        _diag_cov_targets = torch.diagonal(
            _cov_targets, dim1=-2, dim2=-1)
        _diag_cov_est_targets = torch.diagonal(
            _cov_est_targets, dim1=-2, dim2=-1)
        _rownorm_cov_targets =\
            _cov_targets / _diag_cov_targets[..., None]
        _rownorm_cov_est_targets =\
            _cov_est_targets / _diag_cov_est_targets[..., None]
        if self.inv_est:
            _cov_ = torch.linalg.solve(
                _rownorm_cov_est_targets.transpose(-2, -1),
                _rownorm_cov_targets.transpose(-2, -1)).transpose(-2, -1)
            _rat_ = _diag_cov_targets / _diag_cov_est_targets
            _tr_ = _rat_.sum(dim=-1)
            _tr_ += weight_ld * torch.diagonal(
                (_cov_ - self.identity_mat) * _rat_[..., None],
                dim1=-2, dim2=-1).sum(dim=-1)
            # print("_tr:", torch.allclose(_tr, _tr_, rtol=1e-3))
            _logdet_ = _rat_.log().sum(dim=-1)
            _logdet_ += weight_ld * torch.linalg.slogdet(_cov_)[-1]
            # print("_logdet:", torch.allclose(_logdet, _logdet_, rtol=1e-3))
        else:
            raise NotImplementedError
        log_det_div = _tr_ - _logdet_ - cov_est_targets.size(-1)
        return log_det_div
    

    def forward(self, est_targets, targets):
                
        est_targets, extra, extra2 = est_targets
        N = est_targets.shape[-1]
                
        z, phi_z = extra

        if self.identity_mat is None:
            self.identity_mat = torch.eye(
                est_targets.size(-1), device=est_targets.device)
        
        cov_mode = ""
        if cov_mode == "cov_avg":
            cov_targets = 0
            for i in range(2*self.half_padding_to_remove):
                _targets = targets[...,i:-2*self.half_padding_to_remove + i]
                _cov_targets = torch.einsum('...i,...j->...ij', _targets, _targets).float()
                cov_targets += _cov_targets
            cov_targets = cov_targets / (2*self.half_padding_to_remove)
        else:
            _targets = targets[
                ..., self.half_padding_to_remove:-self.half_padding_to_remove]
            cov_targets = torch.einsum('...i,...j->...ij', _targets, _targets).float()
            
        # cov_targets = cov_targets + self.identity_mat *\
        #     torch.clamp(self.EPS * torch.linalg.norm(
        #         cov_targets, 'fro', dim=(-2, -1), keepdim=True), min=self.EPS)
        # cov_est_targets = est_targets + self.identity_mat *\
        #     torch.clamp(self.EPS * torch.linalg.norm(
        #         est_targets, 'fro', dim=(-2, -1), keepdim=True), min=self.EPS)
        cov_targets = cov_targets + self.identity_mat * self.EPS
        cov_est_targets = est_targets + self.identity_mat * self.EPS
            
        current_epoch = self.trainer.current_epoch
        
        ''' Pseudo-curriculum learning with masking '''
        if 'mask' in self.loss_terms:
            N = est_targets.size(-1)
            
            hard_masking = False
            rbf_masking = True
            
            full_cov_after_epoch = self.trainer.max_epochs // 2
            if not self.testing and current_epoch < full_cov_after_epoch:
                if hard_masking:
                    update_every = 5
                    masking_order = max(min(130 * max((current_epoch - self.extra_warmup_loss_diag) // update_every, 0), N), 1) # number of considered diagonals (1 = main diagonal only)
                    if self.testing:
                        masking_order = N
                    row = np.array(masking_order*[1] + [0]*(N-masking_order))
                
                elif rbf_masking:
                    support = np.arange(N)
                    var = 0.1 * N * ((current_epoch//5) + 1)
                    row = np.exp((-support**2 ) / (var**2))
                    masking_order = var
                    
                else:
                    row = torch.ones(N)
                    masking_order = N
                    
                toep = sp.linalg.toeplitz(row)
                mask = torch.from_numpy(toep).to(est_targets.device)
                mask = torch.where(torch.abs(mask) < self.EPS, 0, mask)
            
            else:
                masking_order = N
                mask = torch.ones(N,N).to(est_targets.device)
        else:
            masking_order = N
            mask = torch.ones(N,N).to(est_targets.device)
            
        cov_est_targets *= mask
        cov_targets *= mask
        
        loss = 0
        
        ''' LogDet Divergence between two matrices '''
        if 'ld_src' in self.loss_terms:
            alpha = 1
            loss_log_det_div_src = self.compute_log_det_div_p(
                cov_est_targets, cov_targets, weight_ld=alpha)
            loss += loss_log_det_div_src
        
        if 'is' in self.loss_terms:
            loss_is = self.compute_itakura_saito_diag(
                cov_est_targets, cov_targets)
            loss += loss_is
        
        _mixture = _targets.sum(dim=1, keepdim=True)
        _est_filters = torch.linalg.solve(
            cov_est_targets.sum(dim=1, keepdim=True).transpose(-2, -1),
            cov_est_targets.transpose(-2, -1)).transpose(-2, -1)
        
        ''' Regularize the kernels of the layers '''
        if 'ld_mix' in self.loss_terms:
            cov_mixture = torch.einsum('...i,...j->...ij', _mixture, _mixture).float()
            cov_mixture = cov_mixture + self.identity_mat * self.EPS
            cov_mixture *= mask
            loss_log_det_div_mix = self.compute_log_det_div_mix(
                cov_est_targets, cov_targets, cov_mixture)
            loss += loss_log_det_div_mix

        
        ''' NLL_Multivariate_Gaussian '''
        if 'nll' in self.loss_terms:
                        
            _posterior_means = torch.einsum(
            '...ij,...j->...i', _est_filters, _mixture)
            _posterior_covs = cov_est_targets - torch.einsum(
                '...ij,...jk->...ik', _est_filters, cov_est_targets)
            _posterior_covs = _posterior_covs + self.identity_mat * self.EPS

            __targets = torch.unsqueeze(_targets, dim=1)
            __posterior_means = torch.unsqueeze(_posterior_means, dim=2)
            __posterior_covs = torch.unsqueeze(_posterior_covs, dim=2)

            _logdet_posterior_covs = torch.linalg.slogdet(__posterior_covs)[-1]
            _inv_posterior_covs = torch.linalg.solve(__posterior_covs, self.identity_mat)
            _diff = __targets - __posterior_means
            _cov_ = ((_diff[..., None] * _inv_posterior_covs).sum(dim=-2) * _diff).sum(dim=-1)

            loss_nll = 0.5 * (_logdet_posterior_covs + _cov_)
            
            loss += loss_nll
        else:
            _inv_posterior_covs = None
        
        if 'kl' in self.loss_terms:
            _tril_posterior_covs, _ =\
                torch.linalg.cholesky_ex(_posterior_covs)    # L0 
            _tril_cov_est_targets, _ =\
                torch.linalg.cholesky_ex(cov_est_targets)    # L1
            _tr_sq_M = torch.diagonal(
                torch.linalg.solve_triangular(
                    _tril_cov_est_targets, _tril_posterior_covs, upper=False),
                dim1=-2, dim2=-1).pow(2).sum(dim=-1)
            _sq_y = torch.linalg.solve_triangular(
                    _tril_cov_est_targets, -1 * _posterior_means[..., None],
                upper=False).pow(2).squeeze().sum(dim=-1)
            _ln_ratio = 2 * torch.sum(
                torch.diagonal(_tril_cov_est_targets, dim1=-2, dim2=-1).log()
                - torch.diagonal(_tril_posterior_covs, dim1=-2, dim2=-1).log(),
                dim=-1)
            _kl_div_ = 0.5 * _tr_sq_M + 0.5 * _sq_y + 0.5 * _ln_ratio
            # if not self.simplified:
            #     _kl_div_ = _kl_div_ - 0.5 * _posterior_covs.size(-1)
            loss_kl = torch.unsqueeze(_kl_div_, dim=2)
            loss += loss_kl
            # print(loss_kl.shape)
            # 1/0
        
        if 'kl_old' in self.loss_terms:
            mdot = lambda A, B : torch.einsum('...ij,...jk->...ik', A, B)
            
            # Covariance of the estimate
            Ck = cov_est_targets
            # aka Wiener Filter
            Wk = _est_filters # = Ck @ inv(Cx)
            # Posterior mean
            muk = torch.einsum(
                 '...ij,...j->...i', Wk, _mixture)
            WkCk = mdot(Wk, Ck)
            # Posterio covs
            pCk = Ck - WkCk + self.identity_mat * self.EPS
            # Covariance of the mix 
            Cx = torch.sum(Ck, dim=1).unsqueeze(1)
            
            # Inverses
            Ck_inv = torch.linalg.solve(cov_est_targets, self.identity_mat)
            pCk_inv = Ck_inv + torch.linalg.solve((Cx - Ck), self.identity_mat)
            
            # Log det
            _logdet_Ck = torch.linalg.slogdet(Ck)[-1]
            _logdet_pCk = torch.linalg.slogdet(pCk_inv)[-1]
            
            # mu_T @ pCk_inv @ mu
            pCkinv_mu = torch.einsum('bkij,bkj->bki', pCk_inv, muk)
            mut_pCkinv_mu = torch.einsum('bkj,bkj->bk', muk, pCkinv_mu)
            
            # trace
            _tr = torch.diagonal(WkCk, dim1=-2, dim2=-1).sum(dim=-1)
            
            # Loss
            loss_kl = torch.mean(0.5 * _tr + _logdet_Ck +  _logdet_pCk + mut_pCkinv_mu, dim=-1)
        
        if 'rnd_mix' in self.loss_terms:
            pass
        
        ''' Regularize latent variable '''
        est_latents = phi_z
        if 'znorm' in self.loss_terms:
            loss_znorm = (torch.norm(est_latents)**2) / N
            loss += loss_znorm
        
        # repam tric
        if 'vae' in self.loss_terms:
            means, lvars = extra2['means'], extra2['lvars']
            loss_vae = 10 * (- 0.5 * torch.sum(1 + lvars - means.pow(2) - lvars.exp(), dim=[-2,-1]))[:,:,None]
            loss += loss_vae
            
            
        # if not self.training and self.exp_dir is not None:
        suffix = f'train_epoch-{current_epoch}' if self.training else f'eval_epoch-{current_epoch}'
        save_name = f"{suffix}_logdet_div.png"
        save_path = os.path.join(self.exp_dir, "figures", save_name)
        if not self.training and self.exp_dir is not None:
            if os.path.exists(save_path):
                _mtime = os.path.getmtime(save_path)
                plot_flag = True if time.time() - _mtime > 300 else False
            else:
                plot_flag = True
                        
            if plot_flag:
                print("\n My LOG:")
                if "ld_src" in self.loss_terms: print("- log_det_div[0]:...:\n", loss_log_det_div_src[0].detach().cpu().numpy())
                if "is" in self.loss_terms:     print("- log_is[0]:........:\n", loss_is[0].detach().cpu().numpy())
                if "ld_mix" in self.loss_terms: print("- log_det_mix.......:", loss_log_det_div_mix[0].detach().cpu().numpy())
                if "nll" in self.loss_terms:    print("- loss neg log like.:\n", loss_nll[0].detach().cpu().numpy().squeeze())
                if "kl" in self.loss_terms:     print("- kl................:\n", loss_kl[0].detach().cpu().numpy().squeeze())
                if "vae" in self.loss_terms:    print("- repam.............:\n", loss_vae[0].detach().cpu().numpy().squeeze())
                if "znorm" in self.loss_terms:  print("- loss_latent.......:", loss_znorm.detach().cpu().numpy())
                if "mask" in self.loss_terms:   print("- masking order.....:", masking_order)
                print("- kernel weight.....:", extra2['w_diag'])
                
                _mixture = _targets.detach().sum(dim=1, keepdim=True)
                
                _est_filters = torch.linalg.solve(
                    cov_est_targets.detach().sum(dim=1, keepdim=True).transpose(-2, -1),
                    cov_est_targets.detach().transpose(-2, -1)).transpose(-2, -1)
                _filters = torch.linalg.solve(
                    cov_targets.detach().sum(dim=1, keepdim=True).transpose(-2, -1),
                    cov_targets.detach().transpose(-2, -1)).transpose(-2, -1)
            
                estimates = torch.einsum('...ij,...j->...i', _est_filters, _mixture)
                _compute_sdr = PairwiseNegSDR("sisdr")
                _negsdrs = _compute_sdr(estimates, _targets)
                _pitlosswrapper = PITLossWrapper(_compute_sdr)
                _min_loss, _batch_indices = _pitlosswrapper.find_best_perm(_negsdrs)
                _reordered = _pitlosswrapper.reorder_source(estimates, _batch_indices)

                """ PLOTs """
                widths = [1, 1, 1, 1, 1, 1]
                heights = [2, 2, 2, 2, 2, 1, 1]
                gs_kw = dict(width_ratios=widths, height_ratios=heights)
                fig, axes = plt.subplots(
                    figsize=(24, 18), ncols=6, nrows=7, constrained_layout=True, gridspec_kw=gs_kw)
                mix_indices = [0, int(est_latents.size(0) - 1)]
                # mix_indices = [0, int(est_latents.size(0) // 2 - 1), int(est_latents.size(0) - 1)]
                src_indices = [0, 1, 2]
                row_idx, col_idx = (0, 0)
                for mix_idx in mix_indices:
                    for src_idx in src_indices:
                        _data = est_latents[mix_idx, _batch_indices[mix_idx][src_idx]].detach().cpu().numpy().T
                        _img = axes[row_idx, col_idx].imshow(
                            _data, origin="lower", aspect="auto",
                            vmin=np.percentile(_data, 2.5), vmax=np.percentile(_data, 97.5))
                        plt.colorbar(_img, ax=axes[row_idx, col_idx])
                        axes[row_idx, col_idx].set_title(
                            "estimate (latent) sample:{} source:{}".format(mix_idx + 1, src_idx + 1))
                        axes[row_idx, col_idx].set_xticks(np.arange(0, _data.shape[1] + 1, _data.shape[1] // 4))
                        col_idx += 1
                row_idx, col_idx = (1, 0)
                for mix_idx in mix_indices:
                    for src_idx in src_indices:
                        _data = cov_est_targets[mix_idx, _batch_indices[mix_idx][src_idx]].detach().cpu().numpy()
                        _img = axes[row_idx, col_idx].imshow(
                            _data, origin="lower", aspect="equal",
                            vmin=np.percentile(_data, 2.5), vmax=np.percentile(_data, 97.5))
                        plt.colorbar(_img, ax=axes[row_idx, col_idx])
                        axes[row_idx, col_idx].set_title(
                            "estimate (cov) sample:{} source:{}".format(mix_idx + 1, src_idx + 1))
                        axes[row_idx, col_idx].set_xticks(np.arange(0, _data.shape[1] + 1, _data.shape[1] // 4))
                        axes[row_idx, col_idx].set_yticks(np.arange(0, _data.shape[0] + 1, _data.shape[0] // 4))
                        col_idx += 1
                row_idx, col_idx = (2, 0)
                for mix_idx in mix_indices:
                    for src_idx in src_indices:
                        _data = cov_targets[mix_idx, src_idx].detach().cpu().numpy()
                        _img = axes[row_idx, col_idx].imshow(
                            _data, origin="lower", aspect="equal",
                            vmin=np.percentile(_data, 2.5), vmax=np.percentile(_data, 97.5))
                        plt.colorbar(_img, ax=axes[row_idx, col_idx])
                        axes[row_idx, col_idx].set_title(
                            "target (cov) sample:{} source:{}".format(mix_idx + 1, src_idx + 1))
                        axes[row_idx, col_idx].set_xticks(np.arange(0, _data.shape[1] + 1, _data.shape[1] // 4))
                        axes[row_idx, col_idx].set_yticks(np.arange(0, _data.shape[0] + 1, _data.shape[0] // 4))
                        col_idx += 1
                row_idx, col_idx = (3, 0)
                for mix_idx in mix_indices:
                    for src_idx in src_indices:
                        _data = _est_filters[mix_idx, _batch_indices[mix_idx][src_idx]].detach().cpu().numpy()
                        _img = axes[row_idx, col_idx].imshow(
                            _data, origin="lower", aspect="equal",
                            vmin=np.percentile(_data, 2.5), vmax=np.percentile(_data, 97.5))
                        plt.colorbar(_img, ax=axes[row_idx, col_idx])
                        axes[row_idx, col_idx].set_title(
                            "estimate (filter) sample:{} source:{}".format(mix_idx + 1, src_idx + 1))
                        axes[row_idx, col_idx].set_xticks(np.arange(0, _data.shape[1] + 1, _data.shape[1] // 4))
                        axes[row_idx, col_idx].set_yticks(np.arange(0, _data.shape[0] + 1, _data.shape[0] // 4))
                        col_idx += 1
                row_idx, col_idx = (4, 0)
                for mix_idx in mix_indices:
                    for src_idx in src_indices:
                        _data = _filters[mix_idx, _batch_indices[mix_idx][src_idx]].detach().cpu().numpy()
                        _img = axes[row_idx, col_idx].imshow(
                            _data, origin="lower", aspect="equal",
                            vmin=np.percentile(_data, 2.5), vmax=np.percentile(_data, 97.5))
                        plt.colorbar(_img, ax=axes[row_idx, col_idx])
                        axes[row_idx, col_idx].set_title(
                            "target (filter) sample:{} source:{}".format(mix_idx + 1, src_idx + 1))
                        axes[row_idx, col_idx].set_xticks(np.arange(0, _data.shape[1] + 1, _data.shape[1] // 4))
                        axes[row_idx, col_idx].set_yticks(np.arange(0, _data.shape[0] + 1, _data.shape[0] // 4))
                        col_idx += 1
                row_idx, col_idx = (5, 0)
                for mix_idx in mix_indices:
                    
                    _mix = _reordered[mix_idx, :, :].sum(0).detach().cpu().numpy()
                    _mix_max = 1.1*np.max(np.abs(_mix))
                    
                    for src_idx in src_indices:
                        _data = _reordered[mix_idx, src_idx].detach().cpu().numpy()
                        _img = axes[row_idx, col_idx].plot(_data)
                        axes[row_idx, col_idx].set_title(
                            "estimate (waveform) sample:{} source:{}\nSI-SDR: {:.2f}".format(
                                mix_idx + 1, src_idx + 1,
                                -1 * _negsdrs[mix_idx, _batch_indices[mix_idx][src_idx], src_idx]))
                        axes[row_idx, col_idx].set_xticks(np.arange(0, _data.shape[0] + 1, _data.shape[0] // 4))
                        axes[row_idx, col_idx].set_ylim([-_mix_max, _mix_max])
                        col_idx += 1
                row_idx, col_idx = (6, 0)
                for mix_idx in mix_indices:
                    for src_idx in src_indices:
                        _data = _targets[mix_idx, src_idx].detach().cpu().numpy()
                        _mix = _targets[mix_idx, :].detach().cpu().numpy()
                        _mix = np.sum(_mix, )
                        _img = axes[row_idx, col_idx].plot(_data)
                        axes[row_idx, col_idx].set_title(
                            "target (waveform) sample:{} source:{}".format(mix_idx + 1, src_idx + 1))
                        axes[row_idx, col_idx].set_xticks(np.arange(0, _data.shape[0] + 1, _data.shape[0] // 4))
                        axes[row_idx, col_idx].set_ylim([-_mix_max, _mix_max])
                        col_idx += 1
                fig.suptitle("{} -- Epoch: {}".format(time.ctime(), self.trainer.current_epoch))
                fig.savefig(save_path)
                plt.close()
                
                # ''' Resolve Permutation '''              
                # mixtures = _targets.sum(dim=1, keepdim=True)
                # _est_filters = torch.linalg.solve(
                #     cov_est_targets.sum(dim=1, keepdim=True).transpose(-2, -1),
                #     cov_est_targets.transpose(-2, -1)
                #     ).transpose(-2, -1)
                # _filters = torch.linalg.solve(
                #     cov_targets.sum(dim=1, keepdim=True).transpose(-2, -1),
                #     cov_targets.transpose(-2, -1)
                #     ).transpose(-2, -1)
                # estimates = torch.einsum('...ij,...j->...i', _est_filters, mixtures)
                # _compute_sdr = PairwiseNegSDR("sisdr")
                # _negsdrs = _compute_sdr(estimates, _targets)
                # _pitlosswrapper = PITLossWrapper(_compute_sdr)
                # _min_loss, _batch_indices = _pitlosswrapper.find_best_perm(_negsdrs)
                # estimates = _pitlosswrapper.reorder_source(estimates, _batch_indices)
                        
                # # plt.figure(figsize=(12,9))
                # # c = 1
                # # for i in [0, -1]:
                # #     for j in [0, 1]:
                # #         plt.subplot(2,2,c)
                # #         est = np.diag(cov_est_targets[i, j, ...].detach().cpu().numpy())
                # #         tgt = np.diag(cov_targets[i, j, ...].detach().cpu().numpy())
                # #         plt.ylim([-1,20])
                # #         plt.plot(tgt, label='true')
                # #         plt.plot(est, label='est')
                # #         plt.legend()
                # #         plt.title(f"PSD for sample:{i} source:{j}")
                # #         c += 1
                # # plt.suptitle(time.ctime())
                # # plt.savefig(os.path.join(self.exp_dir, "figures", f"{suffix}_amplitude.png"))
                # # plt.legend()
                # # plt.close()              
                
                # # widths = [1, 1, 1, 1]
                # # heights = [2, 2, 1, 1]
                # # gs_kw = dict(width_ratios=widths, height_ratios=heights)
                # # fig, axes = plt.subplots(
                # #     figsize=(18, 12), ncols=4, nrows=4, constrained_layout=True, gridspec_kw=gs_kw)
                                
                # # c = 0
                # # for i in [0, -1]:
                # #     for j in [0, 1]:
                
                # #         est = _est_filters[i, j, ...].detach().cpu().numpy()
                # #         vmin = np.percentile(est, 2.5)
                # #         vmax = np.percentile(est, 97.5)
                # #         _img = axes[0, c].imshow(est, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
                # #         plt.colorbar(_img, ax=axes[0, c])
                # #         axes[0, c].set_title(f"{c} estimate WF: sample:{i} source:{j}")
                        
                # #         tgt = _filters[i, j, ...].detach().cpu().numpy()
                # #         vmin = np.percentile(tgt, 2.5)
                # #         vmax = np.percentile(tgt, 97.5)
                # #         _img = axes[1, c].imshow(tgt, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
                # #         plt.colorbar(_img, ax=axes[1, c])
                # #         axes[1, c].set_title(f"{c} target WF: sample:{i} source:{j}")
                        
                # #         c += 1
                
                # # _img = axes[2, 0].plot(estimates[0, 0].detach().cpu().numpy())
                # # axes[2, 0].set_title("estimate (waveform) sample:1 source:1")
                # # _img = axes[2, 1].plot(estimates[0, 1].detach().cpu().numpy())
                # # axes[2, 1].set_title("estimate (waveform) sample:1 source:2")
                # # _img = axes[2, 2].plot(estimates[-1, 0].detach().cpu().numpy())
                # # axes[2, 2].set_title("estimate (waveform) sample:2 source:1")
                # # _img = axes[2, 3].plot(estimates[-1, 1].detach().cpu().numpy())
                # # axes[2, 3].set_title("estimate (waveform) sample:2 source:2")
                # # _img = axes[3, 0].plot(_targets[0, 0].detach().cpu().numpy())
                # # axes[3, 0].set_title("target (waveform) sample:1 source:1")
                # # _img = axes[3, 1].plot(_targets[0, 1].detach().cpu().numpy())
                # # axes[3, 1].set_title("target (waveform) sample:1 source:2")
                # # _img = axes[3, 2].plot(_targets[-1, 0].detach().cpu().numpy())
                # # axes[3, 2].set_title("target (waveform) sample:2 source:1")
                # # _img = axes[3, 3].plot(_targets[-1, 1].detach().cpu().numpy())
                # # axes[3, 3].set_title("target (waveform) sample:2 source:2")
                # # fig.suptitle(time.ctime())
                # # fig.savefig(os.path.join(self.exp_dir, "figures", f"{suffix}_wiener.png"))
                # # plt.close()
                                
                
                # widths = [1, 1, 1, 1]
                # heights = [2, 2, 1, 1]
                # gs_kw = dict(width_ratios=widths, height_ratios=heights)
                # fig, axes = plt.subplots(
                #     figsize=(18, 12), ncols=4, nrows=4, constrained_layout=True, gridspec_kw=gs_kw)
                
                # c = 0
                # for i in [0, -1]:
                #     for j in [0, 1]:
                
                #         est = cov_est_targets[i, j, ...].detach().cpu().numpy()
                #         vmin = np.percentile(est, 2.5)
                #         vmax = np.percentile(est, 97.5)
                #         _img = axes[0, c].imshow(est, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
                #         plt.colorbar(_img, ax=axes[0, c])
                #         axes[0, c].set_title(f"estimate (cov) sample:{i} source:{j}")
                        
                #         tgt = cov_targets[i, j, ...].detach().cpu().numpy()
                #         vmin = np.percentile(tgt, 2.5)
                #         vmax = np.percentile(tgt, 97.5)
                #         _img = axes[1, c].imshow(tgt, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
                #         plt.colorbar(_img, ax=axes[1, c])
                #         axes[1, c].set_title(f"target (cov) sample:{i} source:{j}")
                        
                #         c += 1
                
                # _img = axes[2, 0].plot(estimates[0, 0].detach().cpu().numpy())
                # axes[2, 0].set_title("estimate (waveform) sample:1 source:1")
                # _img = axes[2, 1].plot(estimates[0, 1].detach().cpu().numpy())
                # axes[2, 1].set_title("estimate (waveform) sample:1 source:2")
                # _img = axes[2, 2].plot(estimates[-1, 0].detach().cpu().numpy())
                # axes[2, 2].set_title("estimate (waveform) sample:2 source:1")
                # _img = axes[2, 3].plot(estimates[-1, 1].detach().cpu().numpy())
                # axes[2, 3].set_title("estimate (waveform) sample:2 source:2")
                # _img = axes[3, 0].plot(_targets[0, 0].detach().cpu().numpy())
                # axes[3, 0].set_title("target (waveform) sample:1 source:1")
                # _img = axes[3, 1].plot(_targets[0, 1].detach().cpu().numpy())
                # axes[3, 1].set_title("target (waveform) sample:1 source:2")
                # _img = axes[3, 2].plot(_targets[-1, 0].detach().cpu().numpy())
                # axes[3, 2].set_title("target (waveform) sample:2 source:1")
                # _img = axes[3, 3].plot(_targets[-1, 1].detach().cpu().numpy())
                # axes[3, 3].set_title("target (waveform) sample:2 source:2")
                # fig.suptitle(time.ctime())
                # fig.savefig(os.path.join(self.exp_dir, "figures", f"{suffix}_logdet_div.png"))
                # plt.close()
                
                
                # (z1, z2), (phi1, phi2) = extra
                
                # widths = [1, 1, 1, 1]
                # heights = [2, 2, 1, 1]
                # gs_kw = dict(width_ratios=widths, height_ratios=heights)
                # fig, axes = plt.subplots(
                #     figsize=(18, 12), ncols=4, nrows=4, constrained_layout=True, gridspec_kw=gs_kw)
                                
                # _img = axes[0,0].imshow(z1[0,0,:,:].detach().cpu().numpy(), origin="lower", aspect="auto")
                # axes[0,0].set_title("latent z1 - sample:1 source:1")
                # plt.colorbar(_img, ax=axes[0,0])
                # _img = axes[0,1].imshow(z2[0,0,:,:].detach().cpu().numpy(), origin="lower", aspect="auto")
                # axes[0,1].set_title("latent z2 - sample:1 source:2")
                # plt.colorbar(_img, ax=axes[0,1])
                # _img = axes[0,2].imshow(z1[-1,0,:,:].detach().cpu().numpy(), origin="lower", aspect="auto")
                # axes[0,2].set_title("latent z1 - sample:2 source:1")
                # plt.colorbar(_img, ax=axes[0,2])
                # _img = axes[0,3].imshow(z2[-1,0,:,:].detach().cpu().numpy(), origin="lower", aspect="auto")
                # axes[0,3].set_title("latent z2 - sample:2 source:2")
                # plt.colorbar(_img, ax=axes[0,3])
                # _img = axes[1,0].imshow(phi1[0,0,:,:].T.detach().cpu().numpy(), origin="lower", aspect="auto")
                # axes[1,0].set_title("Phi(z1) sample:1 source:1")
                # plt.colorbar(_img, ax=axes[1,0])
                # _img = axes[1,1].imshow(phi2[0,0,:,:].T.detach().cpu().numpy(), origin="lower", aspect="auto")
                # axes[1,1].set_title("Phi(z2) sample:1 source:2")
                # plt.colorbar(_img, ax=axes[1,1])
                # _img = axes[1,2].imshow(phi1[-1,0,:,:].T.detach().cpu().numpy(), origin="lower", aspect="auto")
                # axes[1,2].set_title("Phi(z1) sample:2 source:1")
                # plt.colorbar(_img, ax=axes[1,2])
                # _img = axes[1,3].imshow(phi2[-1,0,:,:].T.detach().cpu().numpy(), origin="lower", aspect="auto")
                # axes[1,3].set_title("Phi(z2) sample:2 source:2")
                # plt.colorbar(_img, ax=axes[1,3])
                
                # _img = axes[2, 0].plot(estimates[0, 0].detach().cpu().numpy())
                # axes[2, 0].set_title("estimate (waveform) sample:1 source:1")
                # _img = axes[2, 1].plot(estimates[0, 1].detach().cpu().numpy())
                # axes[2, 1].set_title("estimate (waveform) sample:1 source:2")
                # _img = axes[2, 2].plot(estimates[-1, 0].detach().cpu().numpy())
                # axes[2, 2].set_title("estimate (waveform) sample:2 source:1")
                # _img = axes[2, 3].plot(estimates[-1, 1].detach().cpu().numpy())
                # axes[2, 3].set_title("estimate (waveform) sample:2 source:2")
                # _img = axes[3, 0].plot(_targets[0, 0].detach().cpu().numpy())
                # axes[3, 0].set_title("target (waveform) sample:1 source:1")
                # _img = axes[3, 1].plot(_targets[0, 1].detach().cpu().numpy())
                # axes[3, 1].set_title("target (waveform) sample:1 source:2")
                # _img = axes[3, 2].plot(_targets[-1, 0].detach().cpu().numpy())
                # axes[3, 2].set_title("target (waveform) sample:2 source:1")
                # _img = axes[3, 3].plot(_targets[-1, 1].detach().cpu().numpy())
                # axes[3, 3].set_title("target (waveform) sample:2 source:2")
                
                # fig.suptitle(time.ctime())
                # fig.savefig(os.path.join(self.exp_dir, "figures", f"{suffix}_latent_var.png"))
                # plt.close()
                                
                # fs = 8000
                # sf.write(os.path.join(self.exp_dir, "s1_est.wav"), estimates[0, 0].detach().cpu().numpy(), fs)
                # sf.write(os.path.join(self.exp_dir, "s2_est.wav"), estimates[0, 1].detach().cpu().numpy(), fs)
                # sf.write(os.path.join(self.exp_dir, "s1_ref.wav"), _targets[0, 0].detach().cpu().numpy(), fs)
                # sf.write(os.path.join(self.exp_dir, "s2_ref.wav"), _targets[0, 1].detach().cpu().numpy(), fs)
                # # sf.write('new_file.flac', data, samplerate)
        return loss
    
    
class PairwiseLogDetDivRank1(_Loss):
    r"""
    Shape:
        - est_targets : :math:`(batch, nsrc, segment_len, segment_len)`.
        - targets: :math:`(batch, nsrc, segment_len)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`. Pairwise losses.
    """
    def __init__(self, padding_to_remove, EPS=1e-6):
        super().__init__()
        self.EPS = EPS
        self.half_padding_to_remove = padding_to_remove // 2
        self.identity_mat = None

    def forward(self, est_targets, targets):
        
        est_targets, extra, extra2 = est_targets
        
        if self.identity_mat is None:
            self.identity_mat = torch.eye(
                est_targets.size(-1), device=est_targets.device)
        _targets = targets[
            ..., self.half_padding_to_remove:-self.half_padding_to_remove]
        cov_targets = torch.einsum('...i,...j->...ij', _targets, _targets)
        cov_targets = cov_targets + self.identity_mat *\
            torch.clamp(self.EPS * torch.linalg.norm(
                cov_targets, 'fro', dim=(-2, -1), keepdim=True), min=self.EPS)
        cov_est_targets = est_targets + self.identity_mat *\
            torch.clamp(self.EPS * torch.linalg.norm(
                est_targets, 'fro', dim=(-2, -1), keepdim=True), min=self.EPS)
        
        est_eigvals, est_eigvecs = torch.linalg.eigh(cov_est_targets)
        
        _est_principal_eigvals = torch.unsqueeze(est_eigvals[..., -1], dim=2)
        _est_principal_eigvecs = torch.unsqueeze(est_eigvecs[..., -1], dim=2)
        __targets = torch.unsqueeze(_targets, dim=1)
        log_det_div =\
            _est_principal_eigvals * torch.sum(
                _est_principal_eigvecs * __targets, dim=-1)**2 -\
            torch.log(_est_principal_eigvals) - 1
            
        if os.path.exists("dummy.png"):
            _mtime = os.path.getmtime("dummy.png")
            plot_flag = True if time.time() - _mtime > 10 else False
        else:
            plot_flag = True
        if plot_flag:
            plt.figure(figsize=(10, 11))
            plt.subplot(321)
            plt.imshow(cov_est_targets[0, 0].detach().cpu().numpy(), origin="lower", aspect="equal")
            plt.colorbar()
            plt.subplot(322)
            plt.imshow(cov_est_targets[0, 1].detach().cpu().numpy(), origin="lower", aspect="equal")
            plt.colorbar()
            plt.subplot(323)
            plt.imshow(cov_targets[0, 0].detach().cpu().numpy(), origin="lower", aspect="equal")
            plt.colorbar()
            plt.subplot(324)
            plt.imshow(cov_targets[0, 1].detach().cpu().numpy(), origin="lower", aspect="equal")
            plt.colorbar()
            plt.subplot(325)
            plt.plot(_targets[0, 0].detach().cpu().numpy())
            plt.subplot(326)
            plt.plot(_targets[0, 1].detach().cpu().numpy())
            plt.tight_layout()
            plt.savefig("dummy.png")
            plt.close()
        return log_det_div
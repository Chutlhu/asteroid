import os
import time
import soundfile as sf
import numpy as np

import torch
from torch.nn.modules.loss import _Loss

import matplotlib.pyplot as plt


class PairwiseLogDetDiv(_Loss):
    r"""
    Shape:
        - est_targets : :math:`(batch, nsrc, segment_len, segment_len)`.
        - targets: :math:`(batch, nsrc, segment_len)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`. Pairwise losses.
    """
    def __init__(self, padding_to_remove, inv_est=True, EPS=1e-6, exp_dir=None):
        super().__init__()
        self.EPS = EPS
        self.inv_est = inv_est
        self.half_padding_to_remove = padding_to_remove // 2
        self.identity_mat = None
        self.check_pairwise_loss_indexing = False
        self.exp_dir = exp_dir

    # vanilla
    def compute_log_det_div(self, cov_est_targets, cov_targets):
        _cov_targets = torch.unsqueeze(cov_targets, dim=1)
        _cov_est_targets = torch.unsqueeze(cov_est_targets, dim=2)
        if self.inv_est:
            # _inv_cov_est_targets = torch.linalg.solve(_cov_est_targets, self.identity_mat)
            _inv_cov_est_targets = torch.linalg.inv(_cov_est_targets)
            _cov = torch.einsum('...ij,...jk->...ik', _cov_targets, _inv_cov_est_targets)
        else:
            # _inv_cov_targets = torch.linalg.solve(_cov_targets, self.identity_mat)
            _inv_cov_targets = torch.linalg.inv(_cov_targets)
            _cov = torch.einsum('...ij,...jk->...ik', _cov_est_targets, _inv_cov_targets)
        _tr = torch.diagonal(_cov, dim1=-2, dim2=-1).sum(dim=-1)
        _sign, _logdet = torch.linalg.slogdet(_cov)
        log_det_div = _tr - _logdet - cov_est_targets.size(-1)
        if not self.check_pairwise_loss_indexing and self.inv_est:
            ''' check based on the code of get_pw_losses(...)
                pair_wise_losses[:, est_idx, target_idx] = loss_func(est_src, target_src, **kwargs) '''
            
            __cov = torch.einsum('...ij,...jk->...ik', cov_targets[:, 1], _inv_cov_est_targets.squeeze()[:, 0])
            __tr = torch.diagonal(__cov, dim1=-2, dim2=-1).sum(dim=-1)
            _, __logdet = torch.linalg.slogdet(__cov)
            _log_det_div = __tr - __logdet - cov_est_targets.size(-1)
            assert torch.allclose(log_det_div[:, 0, 1].detach(), _log_det_div.detach())
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

    def compute_log_det_div_copy(self, cov_est_targets, cov_targets, weight_ld=1.0):
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
        # assert torch.allclose(
        #     _cov_targets,
        #     torch.diag_embed(_diag_cov_targets) @ _rownorm_cov_targets)
        # assert torch.allclose(
        #     _cov_est_targets,
        #     torch.diag_embed(_diag_cov_est_targets) @ _rownorm_cov_est_targets)
        if self.inv_est:
            _inv_rownorm_cov_est_targets = torch.linalg.inv(
                _rownorm_cov_est_targets)
            _cov_ = torch.einsum(
                '...ij,...jk->...ik',
                _rownorm_cov_targets, _inv_rownorm_cov_est_targets)
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
        
        est_targets, extra = est_targets
        
        if self.identity_mat is None:
            self.identity_mat = torch.eye(
                est_targets.size(-1), device=est_targets.device)
        _targets = targets[
            ..., self.half_padding_to_remove:-self.half_padding_to_remove]
        cov_targets = torch.einsum('...i,...j->...ij', _targets, _targets).float()
        cov_targets = cov_targets + self.identity_mat *\
            torch.clamp(self.EPS * torch.linalg.norm(
                cov_targets, 'fro', dim=(-2, -1), keepdim=True), min=self.EPS)
        cov_est_targets = est_targets + self.identity_mat *\
            torch.clamp(self.EPS * torch.linalg.norm(
                est_targets, 'fro', dim=(-2, -1), keepdim=True), min=self.EPS)
        # cov_targets = cov_targets + self.identity_mat * self.EPS
        # cov_est_targets = est_targets + self.identity_mat * self.EPS

        ''' LogDet Divergence between two matrices '''
        log_det_div = self.compute_log_det_div(cov_est_targets, cov_targets)
        # log_det_div = self.compute_log_det_div_copy(cov_est_targets, cov_targets, weight_ld=0.0)
        # log_det_div = self.compute_log_det_div_copy(cov_est_targets, cov_targets, weight_ld=0.2)
        # log_det_div = self.compute_log_det_div_copy(cov_est_targets, cov_targets, weight_ld=0.3)
        # log_det_div = self.compute_log_det_div_copy(cov_est_targets, cov_targets, weight_ld=0.5)
        # log_det_div = self.compute_log_det_div_copy(cov_est_targets, cov_targets, weight_ld=0.7)
        # log_det_div = self.compute_log_det_div_copy(cov_est_targets, cov_targets, weight_ld=1.0)
        ''' Itakura-Saito Divergence between the diagonal entries of two matrices '''
        # log_det_div = self.compute_log_det_div(
        #     torch.diag_embed(torch.diagonal(cov_est_targets, dim1=-2, dim2=-1)),
        #     torch.diag_embed(torch.diagonal(cov_targets, dim1=-2, dim2=-1)))
        # log_det_div = self.compute_itakura_saito_diag(cov_est_targets, cov_targets)
        ''' Combination '''
        # log_det_div = self.compute_log_det_div(cov_est_targets, cov_targets) + self.compute_itakura_saito_diag(cov_est_targets, cov_targets)

        if not self.training and self.exp_dir is not None:
            if os.path.exists(os.path.join(self.exp_dir, "dummy_logdet_div.png")):
                _mtime = os.path.getmtime(os.path.join(self.exp_dir, "dummy_logdet_div.png"))
                plot_flag = True if time.time() - _mtime > 15 else False
            else:
                plot_flag = True
            if plot_flag:
                print("\nlog_det_div[0]:", log_det_div[0].detach().cpu().numpy())
                
                mixtures = _targets.sum(dim=1, keepdim=True)
                
                inv_sum_cov_est_targets = torch.linalg.inv(cov_est_targets.sum(dim=1, keepdim=True))
                filters = torch.einsum('...ij,...jk->...ik', cov_est_targets, inv_sum_cov_est_targets)
                estimates = torch.einsum('...ij,...j->...i', filters, mixtures)
            
                inv_sum_cov_targets = torch.linalg.inv(cov_targets.sum(dim=1, keepdim=True))
                filters_targets = torch.einsum('...ij,...jk->...ik', cov_targets, inv_sum_cov_targets)
             
                # fig = plt.figure(figsize=(12,12))
                # vmin = np.percentile(est, 2.5)
                # vmax = np.percentile(est, 97.5)
                # plt.imshow(est, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
                # plt.colorbar()
                # plt.subplot(122)
                # tgt = filters_targets.sum(dim=1)[0,...].detach().cpu().numpy()
                # vmin = np.percentile(tgt, 2.5)
                # vmax = np.percentile(tgt, 97.5)
                # plt.imshow(tgt, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
                # plt.colorbar()
                # fig.savefig(os.path.join(self.exp_dir, "dummy_sum_wf.png"))
                # plt.close()
                
                widths = [1, 1, 1, 1]
                heights = [2, 2, 1, 1]
                gs_kw = dict(width_ratios=widths, height_ratios=heights)
                fig, axes = plt.subplots(
                    figsize=(18, 12), ncols=4, nrows=4, constrained_layout=True, gridspec_kw=gs_kw)
                                
                c = 0
                for i in [0, -1]:
                    for j in [0, 1]:
                
                        est = filters[i, j, ...].detach().cpu().numpy()
                        vmin = np.percentile(est, 2.5)
                        vmax = np.percentile(est, 97.5)
                        _img = axes[0, c].imshow(est, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
                        plt.colorbar(_img, ax=axes[0, c])
                        axes[0, c].set_title(f"{c} estimate WF: sample:{i} source:{j}")
                        
                        tgt = filters_targets[i, j, ...].detach().cpu().numpy()
                        vmin = np.percentile(tgt, 2.5)
                        vmax = np.percentile(tgt, 97.5)
                        _img = axes[1, c].imshow(tgt, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
                        plt.colorbar(_img, ax=axes[1, c])
                        axes[1, c].set_title(f"{c} target WF: sample:{i} source:{j}")
                        
                        c += 1
                
                _img = axes[2, 0].plot(estimates[0, 0].detach().cpu().numpy())
                axes[2, 0].set_title("estimate (waveform) sample:1 source:1")
                _img = axes[2, 1].plot(estimates[0, 1].detach().cpu().numpy())
                axes[2, 1].set_title("estimate (waveform) sample:1 source:2")
                _img = axes[2, 2].plot(estimates[-1, 0].detach().cpu().numpy())
                axes[2, 2].set_title("estimate (waveform) sample:2 source:1")
                _img = axes[2, 3].plot(estimates[-1, 1].detach().cpu().numpy())
                axes[2, 3].set_title("estimate (waveform) sample:2 source:2")
                _img = axes[3, 0].plot(_targets[0, 0].detach().cpu().numpy())
                axes[3, 0].set_title("target (waveform) sample:1 source:1")
                _img = axes[3, 1].plot(_targets[0, 1].detach().cpu().numpy())
                axes[3, 1].set_title("target (waveform) sample:1 source:2")
                _img = axes[3, 2].plot(_targets[-1, 0].detach().cpu().numpy())
                axes[3, 2].set_title("target (waveform) sample:2 source:1")
                _img = axes[3, 3].plot(_targets[-1, 1].detach().cpu().numpy())
                axes[3, 3].set_title("target (waveform) sample:2 source:2")
                fig.suptitle(time.ctime())
                fig.savefig(os.path.join(self.exp_dir, "dummy_wiener.png"))
                plt.close()
                                
                widths = [1, 1, 1, 1]
                heights = [2, 2, 1, 1]
                gs_kw = dict(width_ratios=widths, height_ratios=heights)
                fig, axes = plt.subplots(
                    figsize=(18, 12), ncols=4, nrows=4, constrained_layout=True, gridspec_kw=gs_kw)
                
                c = 0
                for i in [0, -1]:
                    for j in [0, 1]:
                
                        est = cov_est_targets[i, j, ...].detach().cpu().numpy()
                        vmin = np.percentile(est, 2.5)
                        vmax = np.percentile(est, 97.5)
                        _img = axes[0, c].imshow(est, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
                        plt.colorbar(_img, ax=axes[0, c])
                        axes[0, c].set_title(f"estimate (cov) sample:{i} source:{j}")
                        
                        tgt = cov_targets[i, j, ...].detach().cpu().numpy()
                        vmin = np.percentile(tgt, 2.5)
                        vmax = np.percentile(tgt, 97.5)
                        _img = axes[1, c].imshow(tgt, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
                        plt.colorbar(_img, ax=axes[1, c])
                        axes[1, c].set_title(f"target (cov) sample:{i} source:{j}")
                        
                        c += 1
                
                _img = axes[2, 0].plot(estimates[0, 0].detach().cpu().numpy())
                axes[2, 0].set_title("estimate (waveform) sample:1 source:1")
                _img = axes[2, 1].plot(estimates[0, 1].detach().cpu().numpy())
                axes[2, 1].set_title("estimate (waveform) sample:1 source:2")
                _img = axes[2, 2].plot(estimates[-1, 0].detach().cpu().numpy())
                axes[2, 2].set_title("estimate (waveform) sample:2 source:1")
                _img = axes[2, 3].plot(estimates[-1, 1].detach().cpu().numpy())
                axes[2, 3].set_title("estimate (waveform) sample:2 source:2")
                _img = axes[3, 0].plot(_targets[0, 0].detach().cpu().numpy())
                axes[3, 0].set_title("target (waveform) sample:1 source:1")
                _img = axes[3, 1].plot(_targets[0, 1].detach().cpu().numpy())
                axes[3, 1].set_title("target (waveform) sample:1 source:2")
                _img = axes[3, 2].plot(_targets[-1, 0].detach().cpu().numpy())
                axes[3, 2].set_title("target (waveform) sample:2 source:1")
                _img = axes[3, 3].plot(_targets[-1, 1].detach().cpu().numpy())
                axes[3, 3].set_title("target (waveform) sample:2 source:2")
                fig.suptitle(time.ctime())
                fig.savefig(os.path.join(self.exp_dir, "dummy_logdet_div.png"))
                plt.close()
                
                
                (z1, z2), (phi1, phi2) = extra
                
                widths = [1, 1, 1, 1]
                heights = [2, 2, 1, 1]
                gs_kw = dict(width_ratios=widths, height_ratios=heights)
                fig, axes = plt.subplots(
                    figsize=(18, 12), ncols=4, nrows=4, constrained_layout=True, gridspec_kw=gs_kw)
                                
                axes[0,0].imshow(z1[0,0,:,:].detach().cpu().numpy(), origin="lower", aspect="auto")
                axes[0,0].set_title("latent z1 - sample:1 source:1")
                axes[0,1].imshow(z2[0,0,:,:].detach().cpu().numpy(), origin="lower", aspect="auto")
                axes[0,1].set_title("latent z2 - sample:1 source:2")
                axes[0,2].imshow(z1[-1,0,:,:].detach().cpu().numpy(), origin="lower", aspect="auto")
                axes[0,2].set_title("latent z1 - sample:2 source:1")
                axes[0,3].imshow(z2[-1,0,:,:].detach().cpu().numpy(), origin="lower", aspect="auto")
                axes[0,3].set_title("latent z2 - sample:2 source:2")
                
                axes[1,0].imshow(phi1[0,0,:,:].T.detach().cpu().numpy(), origin="lower", aspect="auto")
                axes[1,0].set_title("Phi(z1) sample:1 source:1")
                axes[1,1].imshow(phi2[0,0,:,:].T.detach().cpu().numpy(), origin="lower", aspect="auto")
                axes[1,1].set_title("Phi(z2) sample:1 source:2")
                axes[1,2].imshow(phi1[-1,0,:,:].T.detach().cpu().numpy(), origin="lower", aspect="auto")
                axes[1,2].set_title("Phi(z1) sample:2 source:1")
                axes[1,3].imshow(phi2[-1,0,:,:].T.detach().cpu().numpy(), origin="lower", aspect="auto")
                axes[1,3].set_title("Phi(z2) sample:2 source:2")
                
                _img = axes[2, 0].plot(estimates[0, 0].detach().cpu().numpy())
                axes[2, 0].set_title("estimate (waveform) sample:1 source:1")
                _img = axes[2, 1].plot(estimates[0, 1].detach().cpu().numpy())
                axes[2, 1].set_title("estimate (waveform) sample:1 source:2")
                _img = axes[2, 2].plot(estimates[-1, 0].detach().cpu().numpy())
                axes[2, 2].set_title("estimate (waveform) sample:2 source:1")
                _img = axes[2, 3].plot(estimates[-1, 1].detach().cpu().numpy())
                axes[2, 3].set_title("estimate (waveform) sample:2 source:2")
                _img = axes[3, 0].plot(_targets[0, 0].detach().cpu().numpy())
                axes[3, 0].set_title("target (waveform) sample:1 source:1")
                _img = axes[3, 1].plot(_targets[0, 1].detach().cpu().numpy())
                axes[3, 1].set_title("target (waveform) sample:1 source:2")
                _img = axes[3, 2].plot(_targets[-1, 0].detach().cpu().numpy())
                axes[3, 2].set_title("target (waveform) sample:2 source:1")
                _img = axes[3, 3].plot(_targets[-1, 1].detach().cpu().numpy())
                axes[3, 3].set_title("target (waveform) sample:2 source:2")
                
                fig.suptitle(time.ctime())
                fig.savefig(os.path.join(self.exp_dir, "dummy_latent_var.png"))
                plt.close()
                                
                # fs = 8000
                # sf.write(os.path.join(self.exp_dir, "s1_est.wav"), estimates[0, 0].detach().cpu().numpy(), fs)
                # sf.write(os.path.join(self.exp_dir, "s2_est.wav"), estimates[0, 1].detach().cpu().numpy(), fs)
                # sf.write(os.path.join(self.exp_dir, "s1_ref.wav"), _targets[0, 0].detach().cpu().numpy(), fs)
                # sf.write(os.path.join(self.exp_dir, "s2_ref.wav"), _targets[0, 1].detach().cpu().numpy(), fs)
                # # sf.write('new_file.flac', data, samplerate)
                
        return log_det_div


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
        __targets = torch.unsqueeze(_targets, dim=1)
        _est_principal_eigvals = torch.unsqueeze(est_eigvals[..., -1], dim=2)
        _est_principal_eigvecs = torch.unsqueeze(est_eigvecs[..., -1], dim=2)
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
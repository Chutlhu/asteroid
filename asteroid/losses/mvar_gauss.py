import os
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from asteroid.losses import PITLossWrapper
from asteroid.losses.sdr import PairwiseNegSDR


class NLL_Multivariate_Gaussian(_Loss):
    r"""
    Shape:
        - est_targets : :math:`(batch, nsrc, segment_len, segment_len)`.
        - targets: :math:`(batch, nsrc, segment_len)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`. Pairwise losses.
    """
    def __init__(self, padding_to_remove, mode="vanilla", simplified=False,
                 inv_est=True, EPS=1e-6, exp_dir=None, use_cupy=False):
        super().__init__()
        self.EPS = EPS
        self.inv_est = inv_est
        self.half_padding_to_remove = padding_to_remove // 2
        self.mode = mode
        self.identity_mat = None
        self.check_pairwise_loss_indexing = False
        self.exp_dir = exp_dir
        self.use_cupy = use_cupy
        self._target_to_plot = None
        self.trainer = None
        self.simplified = simplified

    def compute_log_det_div(self, cov_est_targets, cov_targets):
        _cov_targets = torch.unsqueeze(cov_targets, dim=1)
        _cov_est_targets = torch.unsqueeze(cov_est_targets, dim=2)
        if self.inv_est:
            if cov_est_targets.size(-1) <= 2048:
                _cov = torch.linalg.solve(
                    _cov_est_targets.transpose(-2, -1),
                    _cov_targets.transpose(-2, -1)).transpose(-2, -1)
            else:
                _shape = list(_cov_est_targets.size())
                _shape[2] = _cov_targets.size(2)
                _cov = torch.empty(_shape, device=_cov_est_targets.device)
                for i, j, k in product(range(_shape[0]), range(_shape[1]), range(_shape[2])):
                    _cov[i, j, k] = torch.linalg.solve(
                        _cov_est_targets[i, j, min(_cov_est_targets.size(2)-1, k)].transpose(-2, -1),
                        _cov_targets[i, min(_cov_targets.size(1)-1, j), k].transpose(-2, -1)).transpose(-2, -1)
        else:
            raise NotImplementedError
        _tr = torch.diagonal(_cov, dim1=-2, dim2=-1).sum(dim=-1)
        if _cov_est_targets.size(-1) <= 2048:
            _est_logdet = torch.linalg.slogdet(_cov_est_targets)[-1]
        else:
            _shape = _cov_est_targets.size()
            _est_logdet = torch.empty(_shape[:3], device=_cov_est_targets.device)
            for i, j, k in product(range(_shape[0]), range(_shape[1]), range(_shape[2])):
                _est_logdet[i, j, k] = torch.linalg.slogdet(_cov_est_targets[i, j, k])[-1]
        # log_det_div = _tr - _logdet + _est_logdet - cov_est_targets.size(-1)
        log_det_div = _tr + _est_logdet
        if not self.simplified:
            if _cov_targets.size(-1) <= 2048:
                _logdet = torch.linalg.slogdet(_cov_targets)[-1]
            else:
                _shape = _cov_targets.size()
                _logdet = torch.empty(_shape[:3], device=_cov_targets.device)
                for i, j, k in product(range(_shape[0]), range(_shape[1]), range(_shape[2])):
                    _logdet[i, j, k] = torch.linalg.slogdet(_cov_targets[i, j, k])[-1]
            log_det_div = log_det_div - _logdet - cov_est_targets.size(-1)
        return log_det_div

    def forward(self, est_targets, targets):
        est_targets, est_latents, kernel_params = est_targets

        if self.identity_mat is None:
            self.identity_mat = torch.eye(
                est_targets.size(-1), device=est_targets.device)
        _targets = targets[
            ..., self.half_padding_to_remove:-self.half_padding_to_remove]
        cov_targets = torch.einsum('...i,...j->...ij', _targets, _targets).float()
        cov_targets = cov_targets + self.identity_mat * self.EPS
        # cov_targets = cov_targets + self.identity_mat *\
        #     torch.clamp(self.EPS * torch.linalg.norm(
        #         cov_targets, 'fro', dim=(-2, -1), keepdim=True), min=self.EPS)
        cov_est_targets = est_targets + self.identity_mat * self.EPS
        # cov_est_targets = est_targets + self.identity_mat *\
        #     torch.clamp(self.EPS * torch.linalg.norm(
        #         est_targets, 'fro', dim=(-2, -1), keepdim=True), min=self.EPS)

        _mixture = _targets.sum(dim=1, keepdim=True)
        if cov_est_targets.size(-1) <= 2048:
            _est_filters = torch.linalg.solve(
                cov_est_targets.sum(dim=1, keepdim=True).transpose(-2, -1),
                cov_est_targets.transpose(-2, -1)).transpose(-2, -1)
        else:
            _shape = cov_est_targets.size()
            _est_filters = torch.empty(_shape, device=cov_est_targets.device)
            _sum_cov_est_targets = cov_est_targets.sum(dim=1)
            for i, j in product(range(_shape[0]), range(_shape[1])):
                _est_filters[i, j] = torch.linalg.solve(
                    _sum_cov_est_targets[i].transpose(-2, -1),
                    cov_est_targets[i, j].transpose(-2, -1)).transpose(-2, -1)
        _posterior_means = torch.einsum(
            '...ij,...j->...i', _est_filters, _mixture)
        _posterior_covs = cov_est_targets - torch.einsum(
            '...ij,...jk->...ik', _est_filters, cov_est_targets)
        _posterior_covs = _posterior_covs + self.identity_mat * self.EPS
        # _posterior_covs = _posterior_covs + self.identity_mat *\
        #     torch.clamp(self.EPS * torch.linalg.norm(
        #         _posterior_covs, 'fro', dim=(-2, -1), keepdim=True), min=self.EPS)

        __targets = torch.unsqueeze(_targets, dim=1)
        __posterior_means = torch.unsqueeze(_posterior_means, dim=2)
        __posterior_covs = torch.unsqueeze(_posterior_covs, dim=2)

        if __posterior_covs.size(-1) <= 2048:
            _logdet_posterior_covs = torch.linalg.slogdet(__posterior_covs)[-1]
            _inv_posterior_covs = torch.linalg.solve(__posterior_covs, self.identity_mat)
        else:
            _shape = __posterior_covs.size()
            _logdet_posterior_covs = torch.empty(_shape[:3], device=__posterior_covs.device)
            _inv_posterior_covs = torch.empty(_shape, device=__posterior_covs.device)
            for i, j, k in product(range(_shape[0]), range(_shape[1]), range(_shape[2])):
                _logdet_posterior_covs[i, j, k] = torch.linalg.slogdet(
                    __posterior_covs[i, j, k])[-1]
                _inv_posterior_covs[i, j, k] = torch.linalg.solve(
                    __posterior_covs[i, j, k], self.identity_mat)
        _diff = __targets - __posterior_means
        _cov_ = ((_diff[..., None] * _inv_posterior_covs).sum(dim=-2) * _diff).sum(dim=-1)

        # nll = 0.5 * (_logdet_posterior_covs + _cov_ + _dim_dependent_term)
        nll = 0.5 * (_logdet_posterior_covs + _cov_)
        if not self.simplified:
            _dim_dependent_term = __posterior_covs.size(-1) * torch.log(torch.tensor(2 * torch.pi))
            nll = nll + 0.5 * _dim_dependent_term

        loss = nll
     
        # if not self.training and self.exp_dir is not None and self.trainer.global_rank == 0:
        save_path = os.path.join(self.exp_dir, 'figures', f"epoch_{self.trainer.current_epoch}_dummy_logdet_div.png")
        if not self.training and self.exp_dir is not None:
            if os.path.exists(save_path):
                _mtime = os.path.getmtime(save_path)
                plot_flag = True if time.time() - _mtime > 120 else False
            else:
                plot_flag = True
                
            # if self._target_to_plot is None:
            #     self._target_to_plot = _targets[0].clone()
            
            # if plot_flag and torch.allclose(_targets[0], self._target_to_plot):
            if plot_flag:
                _mixture = _targets.detach().sum(dim=1, keepdim=True)
                if cov_est_targets.size(-1) <= 2048:
                    _est_filters = torch.linalg.solve(
                        cov_est_targets.detach().sum(dim=1, keepdim=True).transpose(-2, -1),
                        cov_est_targets.detach().transpose(-2, -1)).transpose(-2, -1)
                    _filters = torch.linalg.solve(
                        cov_targets.detach().sum(dim=1, keepdim=True).transpose(-2, -1),
                        cov_targets.detach().transpose(-2, -1)).transpose(-2, -1)
                else:
                    _shape = cov_est_targets.size()
                    _est_filters = torch.empty(_shape, device=cov_est_targets.device)
                    _filters = torch.empty(_shape, device=cov_targets.device)
                    _sum_cov_est_targets = cov_est_targets.detach().sum(dim=1)
                    _sum_cov_targets = cov_targets.detach().sum(dim=1)
                    for i, j in product(range(_shape[0]), range(_shape[1])):
                        _est_filters[i, j] = torch.linalg.solve(
                            _sum_cov_est_targets[i].transpose(-2, -1),
                            cov_est_targets[i, j].detach().transpose(-2, -1)).transpose(-2, -1)
                        _filters[i, j] = torch.linalg.solve(
                            _sum_cov_targets[i].transpose(-2, -1),
                            cov_targets[i, j].detach().transpose(-2, -1)).transpose(-2, -1)
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
                mix_indices = [0, int(est_latents.size(0) // 2 - 1), int(est_latents.size(0) - 1)]
                src_indices = [0, 1]
                row_idx, col_idx = (0, 0)
                for mix_idx in mix_indices:
                    for src_idx in src_indices:
                        _data = est_latents[mix_idx, _batch_indices[mix_idx][src_idx]].detach().cpu().numpy()
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
                    for src_idx in src_indices:
                        _data = _reordered[mix_idx, src_idx].detach().cpu().numpy()
                        _img = axes[row_idx, col_idx].plot(_data)
                        axes[row_idx, col_idx].set_title(
                            "estimate (waveform) sample:{} source:{}\nSI-SDR: {:.2f}".format(
                                mix_idx + 1, src_idx + 1,
                                -1 * _negsdrs[mix_idx, _batch_indices[mix_idx][src_idx], src_idx]))
                        axes[row_idx, col_idx].set_xticks(np.arange(0, _data.shape[0] + 1, _data.shape[0] // 4))
                        col_idx += 1
                row_idx, col_idx = (6, 0)
                for mix_idx in mix_indices:
                    for src_idx in src_indices:
                        _data = _targets[mix_idx, src_idx].detach().cpu().numpy()
                        _img = axes[row_idx, col_idx].plot(_data)
                        axes[row_idx, col_idx].set_title(
                            "target (waveform) sample:{} source:{}".format(mix_idx + 1, src_idx + 1))
                        axes[row_idx, col_idx].set_xticks(np.arange(0, _data.shape[0] + 1, _data.shape[0] // 4))
                        col_idx += 1
                fig.suptitle("{} -- Epoch: {}".format(time.ctime(), self.trainer.current_epoch))
                fig.savefig(save_path)
                plt.close()

        return loss
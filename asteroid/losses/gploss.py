import torch
from torch.nn.modules.loss import _Loss

import matplotlib.pyplot as plt


class GPLoss(_Loss):

    def __init__(self, exp_dir, reduction="none", EPS=1e-8):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)
        
        self.EPS = EPS
        self.exp_dir = exp_dir
        

    def forward(self, est_target, target):
        # if target.size() != est_target.size() or target.ndim != 2:
        #     raise TypeError(
        #         f"Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead"
        #     )
            
        print(est_target.shape)
        print(target.shape) 
        
        C1, C2 = est_target.split(1, dim=1)
        plt.imshow(C1[0,0,:,:])
        plt.savefig(self.exp_dir + 'fig_c1.png')
        plt.close()
            
        losses = 0
        
        losses = losses.mean() if self.reduction == "mean" else losses
        return losses
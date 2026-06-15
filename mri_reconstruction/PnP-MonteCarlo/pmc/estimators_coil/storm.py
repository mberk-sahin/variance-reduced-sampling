
import sys 
import torch as th
from abc import ABC, abstractmethod
from .base import EstimatorBase


class STORM(EstimatorBase):
    '''
    STORM optimization algorithm for estimating the score of 
    likelihood term. 
    '''
    def __init__(self, batch_size: int, beta: float, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)

        self.beta = beta
        print(
            "-"*20
            + f"\nEstimator name: STORM"
            + f"\nbatch size: {self.batch_size}"
            + f"\nbeta: {self.beta}"
            + "\n" + "-"*20
        )

    def likelihood_update(
        self,
        x: th.Tensor,  # (B, 1, H, W)
        y: th.Tensor, # (B, C, H, W)
        df_grad_est: th.Tensor | None,
        xprev: th.Tensor, # (B, 1, H, W)
        **kwargs
    ) -> th.Tensor:

        nparticles, num_coils = x.shape[0], self.forward_model.num_coils
        # use all the coils at first step 
        if df_grad_est is None:    
            coil_idxs = th.arange(num_coils, device=x.device).unsqueeze(dim=0).repeat(nparticles, 1)
            new_df_grad_est = self.forward_model.grad(x, y, coil_idxs)
        # sample a subset of coils
        else:
            coil_idxs = self.sample_coil(nparticles, self.batch_size, num_coils)
            # calculate the gradients sequentially
            grad_x = self.forward_model.grad(x, y, coil_idxs)
            grad_xprev = self.forward_model.grad(xprev, y, coil_idxs)
            new_df_grad_est = grad_x + self.beta * (df_grad_est - grad_xprev)
        return new_df_grad_est

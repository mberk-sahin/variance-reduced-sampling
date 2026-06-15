
import sys 
import torch as th
from abc import ABC, abstractmethod
from .base import EstimatorBase


class SGDm(EstimatorBase):
    '''
    Stochasic gradient descent with momentum (SGDm) for estimating the score of 
    likelihood term. 
    '''
    def __init__(self, batch_size: int, beta: float, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)

        self.beta = beta
        print(
            "-"*20
            + f"\nEstimator name: SGDm"
            + f"\nbatch size: {self.batch_size}"
            + f"\nbeta: {self.beta}"
            + "\n" + "-"*20
        )

    def likelihood_update(
        self,
        x: th.Tensor,  # (B, 1, H, W)
        y: th.Tensor, # (B, C, H, W)
        df_grad_est: th.Tensor | None,
        **kwargs
    ) -> th.Tensor:

        nparticles, num_coils = x.shape[0], self.forward_model.num_coils
        
        if df_grad_est is None:    
            # use all the coils as an initialization 
            coil_idxs = th.arange(num_coils, device=x.device).unsqueeze(dim=0).repeat(nparticles, 1) # (nparticles, num_coils)
            new_df_grad_est = self.forward_model.grad(x, y, coil_idxs)
        else:
            # sample a subset of coils
            coil_idxs = self.sample_coil(nparticles, self.batch_size, num_coils)
            grad = self.forward_model.grad(x, y, coil_idxs)
            new_df_grad_est = self.beta * df_grad_est + (1 - self.beta) * grad
        return new_df_grad_est
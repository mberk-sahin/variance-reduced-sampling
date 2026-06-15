
import sys 
import torch as th
from abc import ABC, abstractmethod
from .base import EstimatorBase


class SGDp(EstimatorBase):
    '''
    Stochasic gradient descent with probabilistic momentum (SGDp) for estimating the score of 
    likelihood term. 
    '''
    def __init__(self, batch_size: int, beta: float, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)

        self.beta = beta
        print(
            "-"*20
            + f"\nEstimator name: SGDp"
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
        # use all the coils at initial step
        if df_grad_est is None:    
            coil_idxs = th.arange(num_coils, device=x.device).unsqueeze(dim=0).repeat(nparticles, 1)
            new_df_grad_est = self.forward_model.grad(x, y, coil_idxs)
        # sample a subset of coils
        else:
            # pick a random number between [0,1] for each particle
            p = th.rand(nparticles)
            new_df_grad_est = th.empty_like(x)
            mask = (p < self.beta)
            # use previous estimate with prob. beta (correction step in PAGE)
            sub_nparticles = mask.sum().item()
            if sub_nparticles > 0:
                new_df_grad_est[mask] = df_grad_est[mask]
            # use large batch size (fresh estimate) with prob. 1-beta (reset step in PAGE)
            if nparticles - sub_nparticles > 0:
                coil_idxs = self.sample_coil(nparticles - sub_nparticles, self.batch_size, num_coils)
                new_df_grad_est[~mask] = self.forward_model.grad(x[~mask], y, coil_idxs)
        return new_df_grad_est

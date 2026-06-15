
import sys 
import torch as th
from abc import ABC, abstractmethod
from .base import EstimatorBase


class PAGE(EstimatorBase):
    '''
    PAGE optimization algorithm for estimating the score of 
    likelihood term. 
    '''
    def __init__(self, batch_size: int, beta: float, mini_batch_size: int, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)

        self.beta = beta
        self.mini_batch_size = mini_batch_size
        print(
            "-"*20
            + f"\nEstimator name: PAGE"
            + f"\nbatch size: {self.batch_size}"
            + f"\nmini batch size: {self.mini_batch_size}"
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

        num_coils = self.forward_model.num_coils
        nparticles = x.shape[0]
        if df_grad_est is None:    
            # use all the coils at first step 
            coil_idxs = th.arange(num_coils, device=x.device).unsqueeze(dim=0).repeat(nparticles, 1)
            new_df_grad_est = self.forward_model.grad(x, y, coil_idxs)
        else:
            # toss a die for each particle
            p = th.rand(nparticles)
            mask = (p < self.beta)
            # initialize new gradient estimate
            new_df_grad_est = th.empty_like(x)
            # get the mask for samples to be used in correction step of PAGE
            correct_nsamples = mask.sum().int()
            
            # PAGE -- correction step
            if correct_nsamples > 0:
                coil_idxs1 = self.sample_coil(correct_nsamples, self.mini_batch_size, num_coils)
                grad_x = self.forward_model.grad(x[mask], y, coil_idxs1)
                grad_xprev = self.forward_model.grad(xprev[mask], y, coil_idxs1)
                new_df_grad_est[mask] = df_grad_est[mask] + grad_x - grad_xprev
            
            # PAGE -- reset step
            if nparticles - correct_nsamples > 0:
                coil_idxs2 = self.sample_coil(nparticles - correct_nsamples, self.batch_size, num_coils)
                new_df_grad_est[~mask] = self.forward_model.grad(x[~mask], y, coil_idxs2)

        return new_df_grad_est


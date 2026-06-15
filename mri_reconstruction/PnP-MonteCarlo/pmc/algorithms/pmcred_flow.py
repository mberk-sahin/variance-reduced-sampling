import torch
from .base import BaseAutoDM
from typing import Optional
import sys; sys.path.append('/scratch/gilbreth/sahinm/LangevinMonteCarlo/StochasticInverse/mri_reconstruction/PnP-MonteCarlo/pmc')
from utils.memory_logger import CUDAMemoryLogger


class PMCRED_FLOW(BaseAutoDM):

    def __init__(
        self, 
        forward_model: object,
        estimator: object,
        score_fn: torch.nn.Module, 
        coeff: object,
        gamma: float, 
        alpha: float,
        sigma: float,
        transform: Optional[torch.nn.Module]= None,
        mem_logger: Optional[CUDAMemoryLogger] = None
    ) -> None:
        super().__init__(forward_model, score_fn, coeff, transform, gamma)
        self.alpha = alpha
        self.sigma = sigma
        self.estimator = estimator
        # if None, disable logger by default
        self.mem_logger = mem_logger if mem_logger is not None else CUDAMemoryLogger(interval=0)
        
    def __call__(self, x, y, t, df_grad_est, xprev):
        drift, df_grad_est, score, xprev = self.drift(x, y, t, df_grad_est, xprev)
        xnextdrift = x + drift
        diffusion = self.diffusion(x, t)
        xnext = xnextdrift + diffusion
        return xnext, xnextdrift, x, drift, score, diffusion, df_grad_est, xprev

    def drift(self, x, y, t, df_grad_est, xprev):
        '''
        The iterate x has the following size 
        [B, C, H, W]
        '''
        # save the initial memory values
        self.mem_logger.begin_step(x, t)

        # get an estimate of the gradient of the forward model with memory calculation
        with self.mem_logger.measure_block("df_grad_est"):
            df_grad_est = self.forward_model.grad(x, y)
            #df_grad_est = self.estimator.likelihood_update(x, y, df_grad_est=df_grad_est, xprev=xprev)

        # copy the previous value of xrecon
        xprev = x.detach().clone().to(x.device)

        if self.transform is not None:
            x = self.transform(x)

        # compute the score
        sigma = self.coeff.score_coeff(self, x, t)
        # calculate the score of the prior
        with self.mem_logger.measure_block("score"):
            if self.alpha == 0:
                score = torch.zeros_like(x)
            else:
                with torch.no_grad():
                    alpha = max(self.alpha * sigma ** 2, 1)
                    score = alpha * self.score_fn(
                                    x,
                                    sigma * torch.ones(x.shape[0])
                                )            
        # combine to get the drift (Note the output of the score_fn is negative score)
        drift = self.gamma*(-df_grad_est + score)

        # save the last memory values
        self.mem_logger.end_step()

        return drift, df_grad_est, score, xprev

    def diffusion(self, x, t):
        return self.coeff.brownian_coeff(self, x, t) * torch.randn_like(x)
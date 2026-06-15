
import sys 
import torch as th
from abc import ABC, abstractmethod
from .base import EstimatorBase


class SGD(EstimatorBase):
    '''
    Stochasic gradient descent (SGD) for estimating the score of 
    likelihood term. 
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print(
            "-"*20
            + f"\nEstimator name: SGD"
            + f"\nbatch size: {self.batch_size}"
            + f"\n" + "-"*20
        )

    def likelihood_update(
        self,
        x: th.Tensor,  # (B, 1, H, W)
        y: th.Tensor, # (B, C, H, W)
        **kwargs
    ) -> th.Tensor:

        nparticles, num_coils = y.shape[0], self.forward_model.num_coils
        coil_idxs = self.sample_coil(nparticles, self.batch_size, num_coils)
        return self.forward_model.grad(x, y, coil_idxs)
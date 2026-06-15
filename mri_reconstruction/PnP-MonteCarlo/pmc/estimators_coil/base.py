import torch as th
from abc import ABC, abstractmethod

class EstimatorBase(ABC):
    
    def __init__(
        self, 
        forward_model,
        batch_size: int,
        **kwargs
    ) -> None:
        self.forward_model = forward_model
        self.batch_size = batch_size
    
    @abstractmethod
    def likelihood_update(self, x: th.Tensor, y: th.Tensor):
        pass

    def sample_coil(self, nparticles: int, batch_size: int, num_coils: int):
        ''' sampe coil without replacement '''
        noise = th.rand(nparticles, num_coils)
        coil_idxs = th.argsort(noise, dim=1)[:,:batch_size]
        return coil_idxs

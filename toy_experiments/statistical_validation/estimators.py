
import math
import torch as th

def negative_score_likelihood(
    y: th.Tensor, x: th.Tensor,
    A: th.Tensor, y_std: float,
    noise: th.Tensor,
    noise_type: str
    ):
    '''calculates the score function of the likelihood (y = Ax + e) for data points, x'''
    residual = y - th.einsum('nx,yx->ny', x, A)
    if noise_type == 'gauss':
        grad = th.einsum('ny,yx->nx', residual, A) / y_std ** 2
    elif noise_type == 'laplace':
        grad = math.sqrt(2) / y_std * th.sign(residual) @ A
    else: 
        raise ValueError(f"Only Laplacian and Gaussian noise modelings are available! '{noise_type}' is undefined!")
    return -grad + noise

def negative_log_likelihood(
    y: th.Tensor, x: th.Tensor,
    A: th.Tensor, y_std: float,
    noise_type: str
    ):
    '''calculates the score function of likelihood via ZO estimation'''
    residual = y - th.einsum('ny,xy->nx', x, A)
    if noise_type == 'gauss':
        potential = (residual.pow_(2).sum(dim=1)) / (2*y_std**2)
    elif noise_type == 'laplace':
        potential = math.sqrt(2) / y_std * residual.abs().sum(dim=1)
    else:
        raise ValueError(f"Only Laplacian and Gaussian noise modelings are available! '{noise_type}' is undefined!")
    return potential


class ZO_SGD(object):
    '''
    Zeroth-order stochasic gradient descent (SGD) for estimating the score of 
    likelihood term. 

    We set the total oracle complexity (TOC) per iteration as 
    TOC = mini_batch_size = 6
    '''

    def __init__(
        self,
        mini_batch_size: int = 6, 
        batch_size: int = 12, 
        p: float = 1.0,
        mu: float = 1e-3
    ):
        print("ZO-SGD estimator is being used with the following parameters:\n"
               f"b = {batch_size}\n"
               f"b' = {mini_batch_size}\n"
               f"p = {p}\n"
               f"mu = {mu}\n"
               "-----------------------------------------------------------"
            )
        # zeroth-order approximation parameters
        self.p = p
        self.mu = mu 
        self.batch_size = batch_size           # b
        self.mini_batch_size = mini_batch_size # b'

    def reset_step(self, y: th.Tensor, x_k: th.Tensor, A: th.Tensor, y_std: float, noise_type: str, std_noise_grad: float, init_step: bool=False):
        nparticles, xdim = x_k.shape
        noise_batch_size = 100 if init_step else self.batch_size
        zo_noise = th.randn(noise_batch_size, nparticles, xdim, device=x_k.device)
        dim_noise = std_noise_grad * th.randn(noise_batch_size, nparticles, xdim, device=x_k.device) # synthetic noise simulating dimensionality variance
        # perturb the data
        xp_k = x_k + self.mu * zo_noise
        # calculate forward evaluations
        f_perturbed = negative_log_likelihood(y, xp_k.reshape(-1, xdim), A, y_std, noise_type).reshape(noise_batch_size, nparticles)
        f = negative_log_likelihood(y, x_k, A, y_std, noise_type)
        # calculate the zeroth-order estimate
        m_k = ((f_perturbed - f).unsqueeze(-1) * zo_noise / self.mu + dim_noise).mean(dim=0)
        return m_k
    
    def correction_step(self, y: th.Tensor, x_k: th.Tensor, x_k_1: th.Tensor, m_k_1: th.Tensor, A: th.Tensor, y_std: float, std_noise_grad: float, noise_type: str):
        nparticles, xdim = x_k.shape
        zo_noise = th.randn(self.mini_batch_size, nparticles, xdim, device=x_k.device)
        dim_noise = std_noise_grad * th.randn(self.mini_batch_size, nparticles, xdim, device=x_k.device) 
        # perturb x_k
        xp_k = x_k.unsqueeze(dim=0) + self.mu * zo_noise
        # calculate forward evaluations
        f_perturbed = negative_log_likelihood(y, xp_k.reshape(-1, xdim), A, y_std, noise_type).reshape(self.mini_batch_size, nparticles)
        f = negative_log_likelihood(y, x_k, A, y_std, noise_type)
        # calculate the zeroth-order estimate at iteration k
        zo1 = ((f_perturbed - f).unsqueeze(-1) * zo_noise / self.mu + dim_noise).mean(dim=0).detach().clone()
        # perturb x_k_1
        xp_k_1 = x_k_1.unsqueeze(dim=0) + self.mu * zo_noise
        f_perturbed_2 = negative_log_likelihood(y, xp_k_1.reshape(-1, xdim), A, y_std, noise_type).reshape(self.mini_batch_size, nparticles)
        f_2 = negative_log_likelihood(y, x_k_1, A, y_std, noise_type)
        zo2 = ((f_perturbed_2 - f_2).unsqueeze(-1) * zo_noise / self.mu + dim_noise).mean(dim=0).detach().clone()
        return m_k_1 + zo1 - zo2

    def likelihood_update(
        self, 
        x_k: th.Tensor, 
        x_k_1: th.Tensor,
        m_k_1: th.Tensor,
        y: th.Tensor,
        A: th.Tensor,
        y_std: float,
        noise_type: str,
        rng: th.Generator,
        std_noise_grad: float,
        **kwargs
    ):

        nparticles, xdim = x_k.shape
        if m_k_1 is None:
            m_k = self.reset_step(y, x_k, A, y_std, noise_type=noise_type, std_noise_grad=std_noise_grad, init_step=True)
        else:
            p_tmp = th.rand(nparticles, device=x_k.device, generator=rng).detach()
            mask_large_batch = (p_tmp < self.p)
            m_k = th.empty_like(x_k)
            # perform reset step
            m_k[mask_large_batch] = self.reset_step(y, x_k[mask_large_batch], A, y_std, std_noise_grad=std_noise_grad, noise_type=noise_type)
            # perform correction step
            m_k[~mask_large_batch] = self.correction_step(
                y, x_k[~mask_large_batch], 
                x_k_1[~mask_large_batch], 
                m_k_1[~mask_large_batch], 
                A, y_std,
                std_noise_grad=std_noise_grad,
                noise_type=noise_type
            )
        return m_k

class SGD(object):
    '''
    Stochasic gradient descent (SGD) for estimating the score of 
    likelihood term. 

    We set the total oracle complexity (TOC) per iteration as 
    TOC = mini_batch_size = 6
    '''

    def __init__(self, mini_batch_size: int = 6):
        self.mini_batch_size = mini_batch_size

        print("SGD estimator is being used with the following parameters:\n"
               f"b = {mini_batch_size}\n"
               "-----------------------------------------------------------"
            )

    def likelihood_update(
        self,
        x_k: th.Tensor,
        y: th.Tensor,
        A: th.Tensor,
        std_noise_grad: float,
        y_std: float,
        noise_type: str,
        rng: th.Generator,
        **kwargs
    ): 
        # sample noise
        noise = std_noise_grad * th.randn((self.mini_batch_size,) + tuple(x_k.shape), device=x_k.device, generator=rng) * std_noise_grad
        # calculate SGD
        score_ll = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
        return score_ll
    
class SGDm(object):
    '''
    Stochasic gradient descent (SGD) with momentum for estimating the score of 
    likelihood term. 

    We set the total oracle complexity (TOC) per iteration as 
    TOC = mini_batch_size = 6
    '''
    def __init__(self, beta: float=0.99, mini_batch_size: int = 6):
        self.beta = beta
        self.init_batch_size = 100 # batch size at initialization
        self.mini_batch_size = mini_batch_size

    def likelihood_update(
        self,
        x_k: th.Tensor,
        m_k_1: th.Tensor | None,
        y: th.Tensor,
        A: th.Tensor,
        std_noise_grad: float,
        y_std: float,
        noise_type: str,
        rng: th.Generator,
        **kwargs
    ): 
        
        if m_k_1 is None:
            # sample noise 
            noise = std_noise_grad * th.randn((self.init_batch_size,) + tuple(x_k.shape), device=x_k.device, generator=rng)
            # calculate the estimate at initial step
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
        else:
            # sample noise
            noise = std_noise_grad * th.randn((self.mini_batch_size,) + tuple(x_k.shape), device=x_k.device, generator=rng)
            # SGDm estimate
            m_k = self.beta * m_k_1 + (1-self.beta) * negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
        return m_k

class STORM(object):
    '''
    Stochasic gradient descent with STORM for estimating
    the score of the likelihood term. 

    We set the total oracle complexity (TOC) per iteration as 
    TOC = mini_batch_size * 2 = 6
    '''
    def __init__(self, beta: float=0.99, mini_batch_size: int=3):
        self.beta = beta
        self.init_batch_size = 100
        self.mini_batch_size = mini_batch_size

    def likelihood_update(
        self,
        x_k: th.Tensor,
        x_k_1: th.Tensor,
        m_k_1: th.Tensor,
        y: th.Tensor,
        A: th.Tensor,
        std_noise_grad: float,
        y_std: float,
        noise_type: str,
        rng: th.Generator,
        **kwargs
    ): 
        if m_k_1 is None:
            # sample noise 
            noise = std_noise_grad * th.randn((self.init_batch_size,) + tuple(x_k.shape), device=x_k.device, generator=rng)
            # calculate the estimate at initial step
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
        else:
            # sample noise  
            noise = std_noise_grad * th.randn((self.mini_batch_size,) + tuple(x_k.shape), device=x_k.device, generator=rng)
            # control variate form of STORM update 
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0) + self.beta * (m_k_1 - negative_score_likelihood(y, x_k_1, A, y_std, noise, noise_type).mean(dim=0))
        return m_k

class SGDe(object):
    '''
    Stochasic gradient descent with epochs (SGDe, or vanilla PAGE) for estimating
    the score of the likelihood term. 

    We set the total oracle complexity (TOC) per iteration as 
    TOC = (1-beta) * batch_size = 6
    '''
    def __init__(self, beta: float=0.4, batch_size: int=10):
        self.beta = beta
        self.batch_size = batch_size # batch size for reset
        self.init_batch_size = 100

    def likelihood_update(
        self,
        x_k: th.Tensor,
        m_k_1: th.Tensor,
        y: th.Tensor,
        A: th.Tensor,
        std_noise_grad: float,
        y_std: float,
        noise_type: str,
        rng: th.Generator,
        **kwargs
    ): 
        nparticles, xdim = x_k.shape
        if m_k_1 is None:
            # sample noise 
            noise = std_noise_grad * th.randn((self.init_batch_size,) + tuple(x_k.shape), device=x_k.device, generator=rng)
            # calculate the estimate at initial step
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
        else:
            p = th.rand((), device=x_k.device, generator=rng).detach().cpu().item()
            p_tmp = th.rand(nparticles, device=x_k.device, generator=rng).detach()
            mask_small_batch = (p_tmp < self.beta)
            m_k = th.empty_like(x_k)
            # use the previous iterate (correction step of PAGE)
            m_k[mask_small_batch] = m_k_1[mask_small_batch]
            # reset the estimator
            x_k_blarge = x_k[~mask_small_batch]
            noise = std_noise_grad * th.randn((self.batch_size,) + tuple(x_k_blarge.shape), device=x_k.device, generator=rng)
            m_k[~mask_small_batch] = negative_score_likelihood(y, x_k_blarge, A, y_std, noise, noise_type).mean(dim=0)
        return m_k

class PAGE(object):
    '''
    Stochasic gradient descent with PAGE for estimating
    the score of the likelihood term. Total oracle complexity 
    is calculated as 

    TOC = (1-p)*2*mini_batch_size + p*batch_size = 6

    where p = 1-beta with unified approach. We chose the following values
    p = 0.02, beta=0.98, batch_size = 100, mini_batch_size = 2
    '''    
    def __init__(self, beta: float=0.98, batch_size: int=100, mini_batch_size: int=2):
        self.beta = beta
        self.batch_size = batch_size # batch size for reset
        self.mini_batch_size = mini_batch_size
        self.init_batch_size = 100

    def likelihood_update(
        self,
        x_k: th.Tensor,
        x_k_1: th.Tensor,
        m_k_1: th.Tensor,
        y: th.Tensor,
        A: th.Tensor,
        std_noise_grad: float,
        y_std: float,
        noise_type: str,
        rng: th.Generator,
        **kwargs
    ): 
        nparticles, xdim = x_k.shape
        if m_k_1 is None:
            # sample noise
            noise = std_noise_grad * th.randn((self.init_batch_size,) + tuple(x_k.shape), device=x_k.device, generator=rng)
            # calculate the estimate at initial step
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
        else:
            p_tmp = th.rand(nparticles, device=x_k.device, generator=rng).detach()
            mask_small_batch = (p_tmp < self.beta)
            m_k = th.empty_like(x_k)

            # get the samples with "small batch size update"
            x_k_bsmall, x_k_1_bsmall = x_k[mask_small_batch], x_k_1[mask_small_batch]
            # get the samples with "large batch size update"
            x_k_blarge = x_k[~mask_small_batch]

            # perform correction step
            noise = std_noise_grad * th.randn((self.mini_batch_size,) + tuple(x_k_bsmall.shape), device=x_k.device, generator=rng)
            correction = negative_score_likelihood(y, x_k_bsmall, A, y_std, noise, noise_type).mean(dim=0) - negative_score_likelihood(y, x_k_1_bsmall, A, y_std, noise, noise_type).mean(dim=0) 
            m_k[mask_small_batch] = m_k_1[mask_small_batch] + correction

            # perform reset step
            noise = std_noise_grad * th.randn((self.batch_size,) + tuple(x_k_blarge.shape), device=x_k.device, generator=rng)
            reset = negative_score_likelihood(y, x_k_blarge, A, y_std, noise, noise_type).mean(dim=0)
            m_k[~mask_small_batch] = reset
        return m_k

class EVE(object):
    '''
    Stochasic gradient descent with EVE for estimating
    the score of the likelihood term. 

    We set the total oracle complexity (TOC) per iteration as 
    TOC = 3 * mini_batch_size = 6
    '''    
    def __init__(self, beta1: float=0.999, beta2: float=-0.999, mini_batch_size: int=2):
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.mini_batch_size = mini_batch_size
        self.init_batch_size = 100

    def likelihood_update(
        self,
        x_k: th.Tensor,
        x_k_1: th.Tensor,
        x_k_2: th.Tensor,
        m_k_1: th.Tensor,
        m_k_2: th.Tensor,
        y: th.Tensor,
        A: th.Tensor,
        std_noise_grad: float,
        y_std: float,
        noise_type: str,
        rng: th.Generator,
        **kwargs
    ): 
        # first two step, we use mini-batch
        if (m_k_1 is None) or (m_k_2 is None):
            # sample noise 
            noise = std_noise_grad * th.randn((self.init_batch_size,) + tuple(x_k.shape), device=x_k.device, generator=rng)
            # calculate the estimate at initial step
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
        else:
            # sample a noise
            noise = std_noise_grad * th.randn((self.mini_batch_size,) + tuple(x_k.shape), device=x_k.device, generator=rng)
            # calculate the EVE update terms
            term1 = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
            term2 = m_k_1 - negative_score_likelihood(y, x_k_1, A, y_std, noise, noise_type).mean(dim=0)
            term3 = m_k_2 - negative_score_likelihood(y, x_k_2, A, y_std, noise, noise_type).mean(dim=0)
            # EVE update
            m_k = term1 + (self.beta1 + self.beta2) * term2 - self.beta1 * self.beta2 * term3
        return m_k

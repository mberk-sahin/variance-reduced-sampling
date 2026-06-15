
import torch as th
import math

def negative_score_likelihood(
    y: th.Tensor, x: th.Tensor, 
    A: th.Tensor, y_std: float,
    noise: th.Tensor | None=None,
    noise_type: str='gauss',
    **kwargs):
    '''calculates the score function of the likelihood (y = Ax + e) for data points, x'''
    if noise is None:
        noise = 0

    # calculate the score given noise type of measurements
    if noise_type == 'gauss':
        tmp = -(y - th.einsum('xz,z->x', x, A)) / y_std ** 2
        grad = -tmp[..., None] * A
    elif noise_type == 'laplace':
        grad = math.sqrt(2) / y_std * th.sign(y - x @ A)[:,None] * A[None,]
    else:
        raise ValueError(f"Available noise modelings are 'gauss' or'laplace'. {noise_type} is undefined!")
    return -grad + noise

def negative_log_likelihood(
    y: th.Tensor, x: th.Tensor, 
    A: th.Tensor, y_std: float, 
    noise_type: str='gauss', # choices: ('gauss', 'laplace')
    **kwargs
):
    '''calculates the negative log-likelihood via ZO estimation given a noise modeling'''
    residual = y - x @ A # (# of samples,)
    if noise_type == 'gauss':
        return residual ** 2 / (2*y_std**2)
    elif noise_type == 'laplace':
        return math.sqrt(2) / y_std * th.abs(residual)
    else:
        raise ValueError(f"Available noise modelings are 'gauss' or'laplace'. {noise_type} is undefined!")

class GD(object):
    def likelihood_update(
        self, 
        x_k: th.Tensor,
        y: th.Tensor,
        A: th.Tensor,
        std_noise_grad: float,
        y_std: float,
        rng: th.Generator,
        **kwargs
    ): 
        score_ll = negative_score_likelihood(y, x_k, A, y_std, noise_type=noise_type)
        return score_ll

class SGD(object):
    '''
    Stochasic gradient descent (SGD) for estimating the score of 
    likelihood term. 

    We set the total oracle complexity (TOC) per iteration as 
    TOC = mini_batch_size = 6
    '''

    def __init__(self, mini_batch_size: int = 6):
        self.mini_batch_size = mini_batch_size
        self.grads_used = []
    
    def reset_grads(self):
        self.grads_used = []

    def likelihood_update(
        self,
        x_k: th.Tensor,
        y: th.Tensor,
        A: th.Tensor,
        std_noise_grad: float,
        y_std: float,
        rng: th.Generator,
        noise_type: str,
        **kwargs
    ): 
        # sample noise
        noise = th.randn((self.mini_batch_size,) + tuple(x_k.shape), generator=rng) * std_noise_grad
        # calculate SGD
        score_ll = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
        # save the number of gradients used 
        self.grads_used.append(self.mini_batch_size)
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
        self.grads_used = []
    
    def reset_grads(self):
        self.grads_used = []

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
            noise = th.randn((self.init_batch_size,) + tuple(x_k.shape), generator=rng) * std_noise_grad
            # calculate the estimate at initial step
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
            self.grads_used.append(self.init_batch_size)
        else:
            # sample noise
            noise = th.randn((self.mini_batch_size,) + tuple(x_k.shape), generator=rng) * std_noise_grad
            # SGDm estimate
            m_k = self.beta * m_k_1 + (1-self.beta) * negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
            self.grads_used.append(self.mini_batch_size)
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
        self.grads_used = []

    def reset_grads(self):
        self.grads_used = []

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
            noise = th.randn((self.init_batch_size,) + tuple(x_k.shape), generator=rng) * std_noise_grad
            # calculate the estimate at initial step
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
            self.grads_used.append(self.init_batch_size)
        else:
            # sample noise  
            noise = th.randn((self.mini_batch_size,) + tuple(x_k.shape), generator=rng) * std_noise_grad
            # control variate form of STORM update 
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0) + self.beta * (m_k_1 - negative_score_likelihood(y, x_k_1, A, y_std, noise, noise_type).mean(dim=0))
            self.grads_used.append(2 * self.mini_batch_size)
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
        self.rng = th.Generator().manual_seed(42)
        self.init_batch_size = 100
        self.grads_used = []
    
    def reset_grads(self):
        self.grads_used = []

    def likelihood_update(
        self,
        x_k: th.Tensor,
        m_k_1: th.Tensor,
        y: th.Tensor,
        A: th.Tensor,
        std_noise_grad: float,
        y_std: float,
        rng: th.Generator,
        noise_type: str,
        **kwargs
    ): 
        if m_k_1 is None:
            # sample noise 
            noise = th.randn((self.init_batch_size,) + tuple(x_k.shape), generator=rng) * std_noise_grad
            # calculate the estimate at initial step
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
            self.grads_used.append(self.init_batch_size)
        else:
            p = th.rand((), generator=self.rng).item()
            # use the previous iterate
            if p < self.beta:
                m_k = m_k_1
                self.grads_used.append(0)
            # reset the estimator
            else:
                # sample noise 
                noise = th.randn((self.batch_size,) + tuple(x_k.shape), generator=rng) * std_noise_grad
                m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0) 
                self.grads_used.append(self.batch_size)
        return m_k

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
        self.zo_rng = th.Generator().manual_seed(42)
        self.dnoise_rng = th.Generator().manual_seed(52)

    def reset_grads(self):
        self.grads_used = []

    def reset_step(self, y: th.Tensor, x_k: th.Tensor, A: th.Tensor, y_std: float, noise_type: str, std_noise_grad: float):
        nparticles, xdim = x_k.shape
        zo_noise = th.randn(self.batch_size, nparticles, xdim, device=x_k.device, generator=self.zo_rng)
        dim_noise = th.randn(self.batch_size, nparticles, xdim, device=x_k.device, generator=self.dnoise_rng)
        # perturb the data
        xp_k = x_k + self.mu * zo_noise
        # calculate forward evaluations
        f_perturbed = negative_log_likelihood(y, xp_k.reshape(-1, xdim), A, y_std, noise_type=noise_type).reshape(self.batch_size, nparticles)
        f = negative_log_likelihood(y, x_k, A, y_std, noise_type=noise_type)
        # calculate the zeroth-order estimate
        m_k = ((f_perturbed - f).unsqueeze(-1) * zo_noise) / self.mu
        # add synthetic dimensionality noise to estimator
        m_k = (m_k + dim_noise * std_noise_grad).mean(dim=0)
        return m_k
    
    def correction_step(self, y: th.Tensor, x_k: th.Tensor, x_k_1: th.Tensor, m_k_1: th.Tensor, A: th.Tensor, y_std: float, noise_type: str, std_noise_grad: float):
        nparticles, xdim = x_k.shape
        zo_noise = th.randn(self.mini_batch_size, nparticles, xdim, device=x_k.device, generator=self.zo_rng)
        dim_noise = th.randn(self.mini_batch_size, nparticles, xdim, device=x_k.device, generator=self.dnoise_rng)
        # perturb x_k
        xp_k = x_k + self.mu * zo_noise
        # calculate forward evaluations
        f_perturbed = negative_log_likelihood(y, xp_k.reshape(-1, xdim), A, y_std, noise_type=noise_type).reshape(self.mini_batch_size, nparticles)
        f = negative_log_likelihood(y, x_k, A, y_std, noise_type=noise_type)
        # calculate the zeroth-order estimate at iteration k
        zo1 = ((f_perturbed - f).unsqueeze(-1) * zo_noise).detach().clone() / self.mu
        # add synthetic high-dimensionality noise to estimator 
        zo1 = (zo1 + dim_noise * std_noise_grad).mean(dim=0)
        # perturb x_k_1
        xp_k_1 = x_k_1 + self.mu * zo_noise
        f_perturbed = negative_log_likelihood(y, xp_k_1.reshape(-1, xdim), A, y_std, noise_type=noise_type).reshape(self.mini_batch_size, nparticles)
        f = negative_log_likelihood(y, x_k_1, A, y_std, noise_type=noise_type)
        zo2 = ((f_perturbed - f).unsqueeze(-1) * zo_noise) / self.mu 
        # add synthetic high-dimensionality noise to estimator
        zo2 = (zo2 + dim_noise * std_noise_grad).mean(dim=0)
        return m_k_1 + zo1 - zo2

    def likelihood_update(
        self, 
        x_k: th.Tensor, 
        x_k_1: th.Tensor,
        m_k_1: th.Tensor,
        y: th.Tensor,
        A: th.Tensor,
        y_std: float,
        rng: th.Generator,
        noise_type: str,
        std_noise_grad: float,
        **kwargs
    ):

        nparticles, xdim = x_k.shape
        if m_k_1 is None:
            m_k = self.reset_step(y, x_k, A, y_std, noise_type=noise_type, std_noise_grad=std_noise_grad)
            #print("m_k.shape:", m_k.shape)
        else:
            p_tmp = th.rand(nparticles, device=x_k.device, generator=rng).detach()
            mask_large_batch = (p_tmp < self.p)
            m_k = th.empty_like(x_k)
            # perform reset step
            m_k[mask_large_batch] = self.reset_step(y, x_k[mask_large_batch], A, y_std, noise_type=noise_type, std_noise_grad=std_noise_grad)
            # perform correction step
            m_k[~mask_large_batch] = self.correction_step(
                y, x_k[~mask_large_batch], 
                x_k_1[~mask_large_batch], 
                m_k_1[~mask_large_batch], 
                A, y_std, noise_type=noise_type,
                std_noise_grad=std_noise_grad
            )
            #print("x_k[large_batch].shape:", x_k[mask_large_batch].shape)
            #print("x_k[~large_batch].shape:", x_k[~mask_large_batch].shape)
            #print("limbas is completed!")
            #mport sys; sys.exit()
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
        self.rng = th.Generator().manual_seed(42)
        self.init_batch_size = 100
        self.grads_used = []
    
    def reset_grads(self):
        self.grads_used = []
    
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
            noise = th.randn((self.init_batch_size,) + tuple(x_k.shape), generator=rng) * std_noise_grad
            # calculate the estimate at initial step
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
            self.grads_used.append(self.init_batch_size)
        else:
            p = th.rand((), generator=self.rng).item()
            if p < self.beta:
                noise = std_noise_grad * th.randn((self.mini_batch_size,) + tuple(x_k.shape), generator=rng)
                m_k = m_k_1 + negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0) - negative_score_likelihood(y, x_k_1, A, y_std, noise).mean(dim=0)
                self.grads_used.append(2 * self.mini_batch_size)
            else:
                # sample noise 
                noise = std_noise_grad * th.randn((self.batch_size,) + tuple(x_k.shape), generator=rng)
                m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
                self.grads_used.append(self.batch_size)
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
        self.grads_used = []
    
    def reset_grads(self):
        self.grads_used = []

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
        rng: th.Generator,
        noise_type: str,
        **kwargs
    ): 
        # first two step, we use mini-batch
        if (m_k_1 is None) or (m_k_2 is None):
            # sample noise 
            noise = th.randn((self.init_batch_size,) + tuple(x_k.shape), generator=rng) * std_noise_grad
            # calculate the estimate at initial step
            m_k = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
            self.grads_used.append(self.init_batch_size)
        else:
            # sample a noise
            noise = th.randn((self.mini_batch_size,) + tuple(x_k.shape), generator=rng) * std_noise_grad
            # calculate the EVE update terms
            term1 = negative_score_likelihood(y, x_k, A, y_std, noise, noise_type).mean(dim=0)
            term2 = m_k_1 - negative_score_likelihood(y, x_k_1, A, y_std, noise, noise_type).mean(dim=0)
            term3 = m_k_2 - negative_score_likelihood(y, x_k_2, A, y_std, noise, noise_type).mean(dim=0)
            # EVE update
            m_k = term1 + (self.beta1 + self.beta2) * term2 - self.beta1 * self.beta2 * term3
            self.grads_used.append(3 * self.mini_batch_size)
        return m_k

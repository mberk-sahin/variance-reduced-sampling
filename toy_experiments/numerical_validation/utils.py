
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Tuple, Dict, Optional, Callable
import math
import numpy as np
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.signal import convolve2d
import numpy as np
import torch

def initialize_balanced_forward_model(mu1, mu2, d, m, target_proj_norm=1.0, seed=None, device='cpu', dtype=torch.float32):
    """
    Initialize A (torch tensor) such that A @ (mu1 - mu2) has a fixed norm (target_proj_norm),
    so both GMM modes are equally distinguishable in the measurement space.

    Args:
        mu1 (Tensor): Mean vector of mode 1 (shape [d])
        mu2 (Tensor): Mean vector of mode 2 (shape [d])
        d (int): Dimension of latent space
        m (int): Number of measurements (rows of A)
        target_proj_norm (float): Desired norm of A (mu1 - mu2)
        seed (int): Optional random seed
        device (str): 'cpu' or 'cuda'
        dtype (torch.dtype): Data type, e.g., torch.float32

    Returns:
        A (Tensor): Initialized forward matrix (shape [m, d])
    """
    if seed is not None:
        torch.manual_seed(seed)

    mu1 = mu1.to(device=device, dtype=dtype)
    mu2 = mu2.to(device=device, dtype=dtype)

    delta = mu1 - mu2
    delta_norm = torch.norm(delta)
    if delta_norm.item() == 0:
        raise ValueError("mu1 and mu2 must be different")

    # Initialize A ~ N(0, 1/d)
    A = torch.randn(m, d, device=device, dtype=dtype) / d**0.5

    # Compute current projection norm
    A_delta = A @ delta
    current_norm = torch.norm(A_delta)

    # Rescale
    A *= (target_proj_norm / current_norm)

    return A

def gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Returns a 2D Gaussian kernel."""
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def conv2d_matrix(kernel: torch.Tensor, image_shape: tuple) -> torch.Tensor:
    """
    Constructs the matrix A such that A @ x_vector = blurred_image_vector
    Assumes zero padding and stride 1.
    """
    C, H, W = image_shape
    kH, kW = kernel.shape
    padH, padW = kH // 2, kW // 2

    A = torch.zeros((C*H*W, C*H*W))

    # For each basis vector e_i, apply convolution to get column i of A
    for i in range(H * W):
        basis = torch.zeros(1, 1, H, W)
        basis.view(-1)[i] = 1.0
        blurred = F.conv2d(basis, kernel.view(1,1,kH,kW), padding=(padH, padW))
        A[:, i] = blurred.view(-1)
    
    return A

def get_gaussian_blur_matrix(image_shape=(1, 8, 8), kernel_size=5, sigma=1.0) -> torch.Tensor:
    kernel = gaussian_kernel_2d(kernel_size, sigma)
    A = conv2d_matrix(kernel, image_shape)
    return A

'''
def get_gaussian_kernel(kernel_size=5, sigma=1.0, device='cpu', dtype=torch.float32):
    """Returns a 2D Gaussian kernel as a PyTorch tensor of shape (1, 1, kH, kW)."""
    ax = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)  # Shape: (1, 1, kH, kW)

def make_gaussian_blur_operator(H, W, kernel_size=5, sigma=1.0, device='cpu', dtype=torch.float32):
    """Returns (A, AT) as functions for (n_samples, H, W) tensors."""
    kernel = get_gaussian_kernel(kernel_size, sigma, device=device, dtype=dtype)

    def A(x):  # x: (n_samples, H, W)
        n_samples = x.shape[0]
        x = x.reshape(-1, 32, 32)
        x = x.unsqueeze(1)  # (n_samples, 1, H, W)
        y = F.conv2d(x, kernel, padding=kernel_size//2)
        return y.squeeze(1).reshape(n_samples, -1) # (n_samples, H, W)

    def AT(y):  # adjoint is the same for symmetric kernel
        y = y.unsqueeze(1)  # (n_samples, 1, H, W)
        x = F.conv2d(y, kernel, padding=kernel_size//2)
        return x.squeeze(1).reshape(1, -1)  # (n_samples, H, W)

    return A, AT
'''

def classify_and_compute_mode_stats(
    samples: torch.Tensor,         # (n_samples, x_dim)
    mode_means: torch.Tensor      # (n_modes, x_dim)
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Classifies samples into two clusters based on normalized distance and angular similarity
    to mode1 and mode2, then computes empirical mean and covariance of each cluster.

    Returns:
        ((mean1, cov1), (mean2, cov2))
    """
    
    # Compute Euclidean distances (unnormalized)
    diffs = samples.unsqueeze(1) - mode_means.unsqueeze(0)     # (n_samples, 2, x_dim)
    euclidean_dist = torch.norm(diffs, dim=2)                  # (n_samples, 2)

    # Compute angle misalignment (1 - cosθ)
    samples_norm = torch.nn.functional.normalize(samples, dim=1)        # (n_samples, x_dim)
    modes_norm = torch.nn.functional.normalize(mode_means, dim=1)       # (2, x_dim)
    cosine_sim = torch.matmul(samples_norm, modes_norm.T)               # (n_samples, 2)
    angle_misalignment = 1 - cosine_sim                                 # (n_samples, 2)

    # Normalize each metric to [0, 1] per sample
    dist_min = euclidean_dist.min(dim=1, keepdim=True)[0]
    dist_max = euclidean_dist.max(dim=1, keepdim=True)[0]
    dist_norm = (euclidean_dist - dist_min) / (dist_max - dist_min + 1e-8)

    angle_min = angle_misalignment.min(dim=1, keepdim=True)[0]
    angle_max = angle_misalignment.max(dim=1, keepdim=True)[0]
    angle_norm = (angle_misalignment - angle_min) / (angle_max - angle_min + 1e-8)

    # Combine scores
    combined_score = dist_norm * 0.5 + angle_norm * 0.5  # shape: (n_samples, 2)

    # Assign to mode with lowest combined score
    assignments = torch.argmin(combined_score, dim=1)  # (n_samples,)

    # Split samples
    samples_mode1 = samples[assignments == 0]
    samples_mode2 = samples[assignments == 1]

    # calculate the mode weights 
    n_mode1 = float(len(samples_mode1))
    n_mode2 = float(len(samples_mode2))
    n_total = n_mode1 + n_mode2
    pis = torch.tensor([n_mode1 / n_total, n_mode2 / n_total], device=samples_mode1.device)

    # Empirical means and covariances
    def empirical_stats(x):
        mean = x.mean(dim=0)
        cov = x.T.cov()
        return mean, cov

    mean1, cov1 = empirical_stats(samples_mode1)
    mean2, cov2 = empirical_stats(samples_mode2)

    print("Number of male LMC samples:", len(samples_mode1))
    print("Number of female LMC samples:", len(samples_mode2))
    
    means = torch.stack([mean1, mean2], dim=0)
    covs = torch.stack([cov1, cov2], dim=0)

    return means, covs, pis

def gmm_mean_and_pixel_variance(pis, mus, covs):
    """
    Compute the overall mean and pixel-wise variance of a Gaussian Mixture Model given the 
    weights, means, and covariance matrices of its modes. 
    
    Args:
        pis (Tensor): shape (K,) - mixture weights
        mus (Tensor): shape (K, D) - means
        covs (Tensor): shape (K, D, D) - full covariances

    Returns:
        mean: (D,) overall mean vector
        var: (D,) pixel-wise variance vector
    """
    K, D = mus.shape
    # Compute overall mean
    mean = torch.sum(pis[:, None] * mus, dim=0)  # shape (D,)
    
    # Compute pixel-wise variance
    var = torch.zeros(D, device=mus.device)
    for k in range(K):
        mu_diff_sq = (mus[k] - mean) ** 2  # shape (D,)
        diag_cov = torch.diagonal(covs[k])  # shape (D,)
        var += pis[k] * (mu_diff_sq + diag_cov)

    return mean, var

def load_gmm_params(stats, p=.4):

    # p is weight of the male
    
    mean_male, cov_male = stats['mean_male'], stats['cov_male']
    mean_female, cov_female = stats['mean_female'], stats['cov_female']
    dim = mean_female.shape[0]
    
    mus = torch.stack([
        mean_male + 2,
        mean_female - 2
    ], dim=0)
    
    covs = torch.stack([
        cov_male,
        cov_female
    ], dim=0)

    #eps = torch.eye(covs.shape[1]) * 0.015
    #covs += eps
    
    inv_covs = torch.linalg.inv(covs)
    pis = torch.tensor([p, 1-p])
    
    return {
        'mus': mus, 
        'covs': covs, 
        'inv_covs': inv_covs, 
        'pis': pis
    }

def sample_from_gmm(
        n_samples: int,
        mus: torch.Tensor, 
        covs: torch.Tensor, 
        pis: torch.Tensor,
        rng: torch.Generator,
        class_idx: Optional[int] = None
    ):
    """
    Samples from a Gaussian Mixture Model using PyTorch.
    
    Args:
        weights: Tensor of shape (K,) with mixture weights.
        means: Tensor of shape (K, D) with means for each component.
        covariances: Tensor of shape (K, D, D) with covariance matrices for each component.
        num_samples: Number of samples to draw.
        device: 'cpu' or 'cuda' for GPU acceleration.
    
    Returns:
        samples: Tensor of shape (num_samples, D)
        component_ids: Tensor of shape (num_samples,) indicating which component each sample came from
    """

    #assert mus.device == covs.device == pis.device == rng.device, "All arguments except 'n_samples' must be on the same device!"

    device = mus.device
    n_modes, x_dim = mus.shape
    
    # Sample component indices according to weights
    if class_idx is not None:
        component_ids = torch.tensor([class_idx,] * n_samples).to(device)
    else:
        component_ids = torch.multinomial(pis, n_samples, replacement=True, generator=rng)

    # Allocate samples
    samples = torch.zeros((n_samples, x_dim), device=device)

    for k in range(n_modes):
        # Indices where component k was selected
        mask = (component_ids == k)
        n_k = mask.sum().item()
        if n_k > 0:
            # RANDOM SEED IS NOT SET SO IT SHOULD BE CALLED CAREFULLY AT EACH EXPERIMENT 
            dist = torch.distributions.MultivariateNormal(mus[k], covs[k])
            samples_k = dist.sample((n_k,))
            samples[mask] = samples_k

    return samples.to(device), component_ids.to(device)

def compute_gmm_posterior_blur(
    A: Callable[[torch.Tensor], torch.Tensor],    # (n,) → (m,)
    At: Callable[[torch.Tensor], torch.Tensor],   # (m,) → (n,)
    y: torch.Tensor,                              # (m,)
    prior_pis: torch.Tensor,                      # (K,)
    prior_mus: torch.Tensor,                      # (K, n)
    prior_covs: torch.Tensor,                     # (K, n, n)
    prior_inv_covs: torch.Tensor,                 # (K, n, n)
    beta: float
) -> Dict:
    K, n = prior_mus.shape
    m = y.shape[0]

    # Compute AᵗA x via At(A(x)) for any x
    def apply_AtA(x):  # (n,) → (n,)
        return At(A(x)) / beta**2  # scaled AtA x

    # Build common AtA matrix approximation (slow but exact)
    # We'll apply AtA to the identity matrix to get the matrix version (n, n)
    I_n = torch.eye(n, device=y.device)
    A_t_A_scaled = torch.stack([apply_AtA(I_n[i, None]) for i in range(n)], dim=1)  # (n, n)
    print("A_t_A_scaled.shape:", A_t_A_scaled.shape)

    posterior_precisions = A_t_A_scaled + prior_inv_covs  # (K, n, n)
    chol = torch.linalg.cholesky(posterior_precisions)
    posterior_covs = torch.cholesky_inverse(chol)

    # Aᵗ y / β²
    Aty = At(y) / beta**2  # (n,)
    rhs = Aty + torch.einsum("kij,kj->ki", prior_inv_covs, prior_mus)
    print("rhs.shape:", rhs.shape)
    print("posterior_covs.shape:", posterior_covs.shape)
    posterior_means = torch.einsum("kij,kj->ki", posterior_covs, rhs)

    # Likelihood term: N(y | A μ_k, A Σ_k Aᵗ + β² I)
    A_mu = torch.stack([A(prior_mus[k]) for k in range(K)])  # (K, m)
    
    # Covariance: A Σ_k Aᵗ for each component
    def A_cov_At(cov_k):  # (n, n) → (m, m)
        I_n = torch.eye(n, device=cov_k.device)
        A_cov = torch.stack([A(cov_k @ I_n[i, None]) for i in range(n)], dim=1)  # (1, m, n)
        A_cov = A_cov.squeeze(dim=0)
        return A_cov @ A_cov.T  # (m, m)

    A_covs_At = torch.stack([A_cov_At(prior_covs[k]) for k in range(K)])  # (K, m, m)

    noise_cov = beta**2 * torch.eye(m, device=y.device)  # (m, m)
    S_k = A_covs_At + noise_cov[None, :, :]  # (K, m, m)
    residuals = y[None, :] - A_mu  # (K, m)

    chol_Sk = torch.linalg.cholesky(S_k)
    alpha = torch.cholesky_solve(residuals.unsqueeze(-1), chol_Sk).squeeze(-1)
    quad_terms = (residuals * alpha).sum(dim=1)

    log_det = 2 * torch.log(torch.diagonal(chol_Sk, dim1=1, dim2=2)).sum(dim=1)
    log_likelihoods = -0.5 * (quad_terms + log_det + m * math.log(2 * torch.pi))
    log_w = torch.log(prior_pis) + log_likelihoods
    posterior_weights = F.softmax(log_w, dim=0)

    return {
        "mus": posterior_means,
        "covs": posterior_covs,
        "inv_covs": posterior_precisions,
        "pis": posterior_weights
    }


def compute_gmm_posterior(
    A: torch.Tensor,                         # shape (m, n)
    y: torch.Tensor,                         # shape (m,)
    prior_pis: torch.Tensor,                 # shape (K,)
    prior_mus: torch.Tensor,                 # shape (K, n)
    prior_covs: torch.Tensor,                # shape (K, n, n)
    prior_inv_covs: torch.Tensor,            # shape (K, n, n)
    beta: float
) -> Dict:
    """
    Vectorized computation of posterior GMM parameters.

    Returns:
        Dict with:
            - posterior_means: (K, n)
            - posterior_covs:  (K, n, n)
            - posterior_weights: (K,)
    """
    K, n = prior_mus.shape
    m = y.shape[0]

    At = A.T                                # (n, m)

    # Compute posterior precisions: AᵀA / β² + Σ⁻¹ (K, n, n)
    A_t_A = At @ A                          # (n, n)
    A_t_A_scaled = A_t_A / beta**2         # (n, n)
    posterior_precisions = A_t_A_scaled[None, :, :] + prior_inv_covs  # (K, n, n)

    # Posterior covariances: (posterior precision)⁻¹ (K, n, n)
    chol = torch.linalg.cholesky(posterior_precisions)                # (K, n, n)
    posterior_covs = torch.cholesky_inverse(chol)                     # (K, n, n)

    # Right-hand side: Aᵀ y / β² + Σ⁻¹ μ (K, n)
    Aty = At @ y                        # (n,)
    Aty_scaled = Aty / beta**2         # (n,)
    rhs = Aty_scaled[None, :] + torch.einsum("kij,kj->ki", prior_inv_covs, prior_mus)  # (K, n)

    # Posterior means: Σ_post × rhs (K, n)
    posterior_means = torch.einsum("kij,kj->ki", posterior_covs, rhs)  # (K, n)
 
    # Compute log likelihoods: log N(y; A μ_k, A Σ_k Aᵀ + β² I)
    # calculate mean
    A_mu = prior_mus @ At              # (K, m)
    mode_diffs = torch.sum((A_mu[0] - A_mu[1]) ** 2)
    #print("mode differences:", mode_diffs.item())
    # calculate covariance
    #A_covs_At = A @ prior_covs @ At.T       # (K, m, m)
    A_covs_At = torch.einsum('ij,kjz,mz->kim', A, prior_covs, A) # (K, m, m)
    noise_cov = beta**2 * torch.eye(m, device=A.device)  # (m, m)
    residuals = y[None, :] - A_mu      # (K, m)
    S_k = A_covs_At + noise_cov[None, :, :]  # (K, m, m)

    # Cholesky decomp of S_k
    chol_Sk = torch.linalg.cholesky(S_k)  # (K, m, m)
    # Solve for quadratic form
    alpha = torch.cholesky_solve(residuals.unsqueeze(-1), chol_Sk).squeeze(-1)  # (K, m)
    quad_terms = (residuals * alpha).sum(dim=1)  # (K,)

    # Log determinant
    log_det = 2 * torch.log(torch.diagonal(chol_Sk, dim1=1, dim2=2)).sum(dim=1)  # (K,)
    log_likelihoods = -0.5 * (quad_terms + log_det + m * math.log(2 * torch.pi))  # (K,)

    # Log unnormalized weights
    log_prior = torch.log(prior_pis)  # (K,)
    #print("log_prior mode1:", log_prior[0].item())
    #print("log_prior mode2", log_prior[1].item())
    #print("-"*20)
    #print("log_likelihoods mode1", log_likelihoods[0].item())
    #print("log_likelihoods mode2", log_likelihoods[1].item())
    log_w = log_prior + log_likelihoods

    # Stable softmax
    posterior_weights = F.softmax(log_w, dim=0)  # (K,)

    return {
        "mus": posterior_means,       # (K, n)
        "covs": posterior_covs,         # (K, n, n)
        "inv_covs": posterior_precisions, # (K, n, n)
        "pis": posterior_weights    # (K,)
    }


def compute_gmm_posterior_v1(
    A: torch.Tensor, 
    y: torch.Tensor, 
    prior_pis: torch.Tensor, 
    prior_mus: torch.Tensor, 
    prior_covs: torch.Tensor, 
    prior_inv_covs: torch.Tensor,
    beta: float
)->Dict:
    """
    Compute the posterior GMM parameters p(x | y) for a linear model y = A x + e, with e ~ N(0, beta^2 * I),
    and a GMM prior over x.

    Args:
        A: Tensor of shape (y_dim, x_dim), the forward operator.
        y: Tensor of shape (y_dim,), the observation.
        prior_pis: Tensor of shape (n_modes,), mixture weights.
        prior_mus: Tensor of shape (n_modes, x_dim), prior means.
        prior_covs: Tensor of shape (n_modes, x_dim, x_dim), prior covariances.
        beta: float, standard deviation of the Gaussian noise.

    Returns:
        posterior_pis: Tensor of shape (n_modes,), posterior mixture weights.
        posterior_mus: Tensor of shape (n_modes, x_dim), posterior means.
        posterior_covs: Tensor of shape (n_modes, x_dim, x_dim), posterior covariances.
    """
    n_modes, x_dim = prior_mus.shape
    y_dim = y.shape[0]
    
    A = A.to(y.device)
    I_M = torch.eye(y_dim, device=y.device)
    

    # Compute A Σ_k A^T for each component (n_modes, y_dim, y_dim)
    ASigma = torch.einsum('md,kde->kme', A, prior_covs)  # (n_modes, y_dim, x_dim)
    cov_y = torch.einsum('kmd,nd->kmn', ASigma, A) + beta**2 * I_M  # (n_modes, y_dim, y_dim)

    # Compute Cholesky for stability
    cov_y_chol = torch.linalg.cholesky(cov_y)
    cov_y_inv = torch.cholesky_inverse(cov_y_chol)  # (n_modes, y_dim, y_dim)

    # Kalman gain: K_k = Σ_k A^T (A Σ_k A^T + β² I)^(-1)
    K_gain = torch.einsum('kmn,dn,kde->kme', prior_covs, A, cov_y_inv) # (n_modes, x_dim, y_dim)


    #Sigma_AT = torch.einsum('kde,md->kme', prior_covs, A.T)  # (K, D, M)
    #K_gain = torch.einsum('kdm,kmn->kdn', Sigma_AT, cov_y_inv)  # (K, D, M)

    # Posterior means: μ'_k = μ_k + K_k (y - A μ_k)
    A_mu = torch.einsum('md,kd->km', A, prior_mus)  # (n_modes, y_dim)
    y_minus_A_mu = y.unsqueeze(0) - A_mu  # (n_modes, y_dim)
    correction = torch.einsum('kdm,km->kd', K_gain, y_minus_A_mu)  # (n_modes, x_dim)
    posterior_mus = prior_mus + correction  # (n_modes, x_dim)

    # Posterior covariances: Σ'_k = Σ_k - K_k A Σ_k
    posterior_covs = prior_covs - torch.einsum('kdm,kme->kde', K_gain, ASigma)  # (K, D, D)

    # Mahalanobis distances
    alpha = torch.einsum('kmn,kn->km', cov_y_inv, y_minus_A_mu)  # (n_modes, y_dim)
    mahal = torch.einsum('km,km->k', y_minus_A_mu, alpha)  # (n_modes,)

    # Log determinant
    log_det = 2 * torch.sum(torch.log(torch.diagonal(cov_y_chol, dim1=-2, dim2=-1)), dim=1)  # (n_modes,)

    # Log-likelihoods
    log_liks = -0.5 * (mahal + log_det + y_dim * torch.log(torch.tensor(2 * torch.pi, device=A.device)))  # (n_modes,)
    log_liks = torch.log(prior_pis) + log_liks  # (n_modes,)
    log_post_pis = log_liks - torch.logsumexp(log_liks, dim=0)
    posterior_pis = torch.exp(log_post_pis)

    return {
        'mus' : posterior_mus,
        'covs': posterior_covs,
        'inv_covs': torch.linalg.inv(posterior_covs),
        'pis': posterior_pis
    }

def bounded_noise(
    size: Tuple[int], 
    rng: torch.Generator, 
    device: str='cuda', 
    epsmax: float=0.0
):
    '''generates a noise to be added to the score prior for synthetic error generation at most eps_max '''
    noise = torch.randn(size=size, generator=rng, device=device)
    norms = torch.linalg.norm(noise, dim=-1, keepdim=True)
    scale = epsmax / (norms + 1e-12)
    #scale = torch.clamp(epsmax / (norms + 1e-12), max=1.0)
    return noise * scale

def log_posterior(
    x: torch.Tensor, 
    y: torch.Tensor, 
    A: torch.Tensor, 
    prior_means: torch.Tensor, 
    prior_covs: torch.Tensor, 
    prior_pis: torch.Tensor,
    beta: float
):
    means = torch.einsum('x,yx->y', A,  grid)
    log_ll = log_likelihood(y, means, beta ** 2, eps=1e-8)
    log_prior = log_prior(x, prior_means, prior_covs, prior_pis)
    return log_prior + log_ll


def log_prior(
    x: torch.Tensor,
    prior_means: torch.Tensor, 
    prior_covs: torch.Tensor, 
    prior_pis: torch.Tensor
)->torch.Tensor:

    # calculate log-prior
    K = gmm_means.shape[0]
    log_probs = []
    for k in range(K):
        mvns = MultivariateNormal(loc=prior_means[k], covariance_matrix=prior_covs[k])
        tmp = mvns.log_prob(x) # (N,)
        log_probs.append(tmp)
    log_probs = torch.stack(log_probs, dim=0)
    log_pis = torch.log(prior_pis + 1e-12).unsqueeze(dim=1) # (K, 1)
    log_prior = torch.logsumexp(log_pis + log_probs, dim=0)
    return log_prior


##################### FUNCTIONS FOR GENERATING USEFUL FORWARD OPERATOR #####################

def generate_informative_A(mus, rng, min_dot=0.5):
    diff = mus[1] - mus[0]
    diff = diff / diff.norm()
    while True:
        A = torch.rand(2, generator=rng, device=rng.device) * 2 - 1
        A = A / A.norm()
        if torch.abs(torch.dot(A, diff)) > min_dot:
            return A

def generate_acceptable_A(mus, covs, rng, min_var=20.0, max_var=200.0):
    """
    Samples a forward operator A such that the induced variance of y = A^T x
    remains within a reasonable range.

    Args:
        mus: (K, D) tensor of GMM means
        covs: (K, D, D) tensor of GMM covariances
        rng: torch.Generator for reproducibility
        min_var: lower bound on induced variance
        max_var: upper bound on induced variance
    """
    K, D = mus.shape
    weights = torch.ones(K, device=rng.device) / K  # assume uniform for now

    while True:
        A = torch.rand(1, D, generator=rng, device=rng.device) * 2 - 1  # Uniform[-1, 1]^D
        A = A / A.norm()

        # Compute mean and variance of A^T x for GMM
        # E[x] = sum_k pi_k mu_k, Cov[x] = sum_k pi_k (cov_k + mu_k mu_k^T) - E[x] E[x]^T
        mean_x = torch.sum(weights[:, None] * mus, dim=0)  # (D,)
        second_moment = torch.sum(weights[:, None, None] * (
            covs + mus[:, :, None] @ mus[:, None, :]), dim=0)
        cov_x = second_moment - mean_x[:, None] @ mean_x[None, :]  # (D, D)

        # Var(y) = A^T Cov[x] A
        var_y = A @ cov_x @ A.T

        if min_var <= var_y.item() <= max_var:
            return A.squeeze(0)

def make_symmetric_A(mu_0: torch.Tensor, mu_1: torch.Tensor, m: int, rng=None) -> torch.Tensor:
    """
    Constructs a measurement matrix A ∈ ℝ^{m × d} whose row span includes the direction
    between mu_0 and mu_1 and is orthonormal overall. This ensures symmetric projection
    of the two modes with respect to their midpoint.

    Args:
        mu_0 (torch.Tensor): Mean vector of mode 0, shape (d,)
        mu_1 (torch.Tensor): Mean vector of mode 1, shape (d,)
        m (int): Number of measurement rows (y_dim)
        rng (torch.Generator, optional): Random number generator for reproducibility

    Returns:
        A (torch.Tensor): Measurement matrix of shape (m, d)
    """
    assert mu_0.shape == mu_1.shape, "mu_0 and mu_1 must have the same shape"
    d = mu_0.numel()
    assert m <= d, "Cannot create orthonormal matrix with more rows than dimensions"

    device = mu_0.device
    dtype = mu_0.dtype

    delta = mu_1 - mu_0
    delta = delta / delta.norm()

    # Initialize with delta
    basis = [delta]

    while len(basis) < m:
        v = torch.rand(d, device=device, dtype=dtype, generator=rng)
        # Orthogonalize against all previous basis vectors
        for b in basis:
            v -= torch.dot(v, b) * b
        norm_v = v.norm()
        if norm_v > 1e-6:
            basis.append(v / norm_v)

    A = torch.stack(basis, dim=0)  # shape (m, d)
    return A

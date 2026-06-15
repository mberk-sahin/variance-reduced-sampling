# imports
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.sparse import lil_matrix, csr_matrix
from typing import Tuple, Dict, Optional, Callable

# torch imports
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

try:
    import scipy.ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _save_with_cbar(img, cmap, save_path, norm=None, min_text_pos=(-0.35, 0.25), max_text_pos=(1.35, 0.25),
                    figsize=(2, 1), add_border=False):
    """
    Helper: save an image with a horizontal colorbar styled like the old plotting code.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis('off')

    im = ax.imshow(img, cmap=cmap, norm=norm)

    # horizontal colorbar under the image
    cbar = plt.colorbar(
        im, ax=ax, orientation='horizontal',
        fraction=0.046, pad=0.03, shrink=0.5, aspect=10
    )
    cbar.set_ticks([])

    # min/max labels (positions match your previous style)
    cbar.ax.text(min_text_pos[0], min_text_pos[1], f"{np.min(img):.2f}", va='center', ha='left',
                 transform=cbar.ax.transAxes, fontsize=5)
    cbar.ax.text(max_text_pos[0], max_text_pos[1], f"{np.max(img):.2f}", va='center', ha='right',
                 transform=cbar.ax.transAxes, fontsize=5)

    # save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_stat_imgs(
        mean: torch.Tensor,
        cov: torch.Tensor,
        save_dir: Path | str,
        *,
        smoothing: bool = False,
        gamma: float = 1.0,
        mean_cmap: str = 'viridis',
        var_cmap: str = 'Blues',
        verbose: bool = True
    ):
    """
    Save mean and variance images with old-style horizontal colorbars and min/max labels.

    - smoothing: Gaussian σ=0.5 on both mean and var (requires SciPy).
    - gamma: variance PowerNorm gamma (gamma<1 brightens, >1 darkens). 1.0 disables.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    H = W = 32
    mean_img = mean.reshape(H, W).detach().cpu().numpy()
    var_img = torch.diagonal(cov).reshape(H, W).detach().cpu().numpy()

    if smoothing and _HAS_SCIPY:
        mean_img = ndi.gaussian_filter(mean_img, sigma=0.5)
        var_img  = ndi.gaussian_filter(var_img,  sigma=0.5)

    # Keep a copy of unclipped variance for min label; clip only for display
    # (visually matches your prior 99th-percentile cap)
    v99 = np.quantile(var_img, 0.99)
    var_disp = np.clip(var_img, None, v99)

    # Power-law normalization for variance if requested
    var_norm = mcolors.PowerNorm(gamma=gamma) if (gamma is not None and gamma != 1.0) else None

    # Stats archive like before
    np.savez(save_dir / "stats.npz", mean=mean_img, var=var_img)

    # Mean image (use same offsets as your original: negatives get slightly wider left label)
    mean_min_offset = -0.35 if np.min(mean_img) < 0 else -0.30
    _save_with_cbar(
        mean_img, mean_cmap, save_dir / "mean.png", norm=None,
        min_text_pos=(mean_min_offset, 0.25), max_text_pos=(1.35 if np.min(mean_img) < 0 else 1.30, 0.25)
    )

    # Variance image (match your old positions: -0.37 / 1.37), with clipping + optional gamma
    _save_with_cbar(
        var_disp, var_cmap, save_dir / "var.png", norm=var_norm,
        min_text_pos=(-0.37, 0.25), max_text_pos=(1.37, 0.25)
    )

    if verbose:
        print(f"Mean and variance images (with styled colorbars) saved to: '{str(save_dir)}'")

def compute_posterior(
    A: torch.Tensor,              # (m, n)
    y: torch.Tensor,              # (m,)
    prior_params: Dict[str, torch.Tensor],
    y_std: float
):
    """
    Compute posterior N(mu_post, cov_post) for:
        y = A x + e,   e ~ N(0, y_std^2 I)
        x ~ N(prior_mu, prior_cov)

    prior_mu: torch.Tensor,       # (n,)
    prior_cov: torch.Tensor,      # (n, n)
    prior_inv_cov: torch.Tensor,  # (n, n)
    """
    prior_mu = prior_params['mean']
    prior_cov = prior_params['cov']
    prior_inv_cov = prior_params['precision']

    At = A.T                            # (n, m)

    # Posterior precision: Σ_post^{-1} = Σ^{-1} + AᵀA / y_std²
    A_t_A = At @ A                      # (n, n)
    posterior_precision = prior_inv_cov + A_t_A / (y_std**2)   # (n, n)

    # Posterior covariance: Σ_post = (Σ_post^{-1})^{-1}
    chol = torch.linalg.cholesky(posterior_precision)          # (n, n)
    posterior_cov = torch.cholesky_inverse(chol)               # (n, n)

    # RHS: Σ^{-1} μ + Aᵀ y / y_std²
    rhs = prior_inv_cov @ prior_mu + (At @ y) / (y_std**2)     # (n,)

    # Posterior mean: μ_post = Σ_post * RHS
    posterior_mean = posterior_cov @ rhs                       # (n,)

    return {
        "mean": posterior_mean,          # (n,)
        "cov": posterior_cov,          # (n, n)
        "inv_cov": posterior_precision # (n, n)
    }


def save_test_img_y(test_img, y, A, save_dir):
    """
    Visualized the test image and measurements in the same figure and save them to 
    save_path. 
    """

    # round the size of the measurements to the larger closes dimensionality

    fig, axes = plt.subplots(1, 2, figsize=(3, 2))
    y = y.numpy().ravel()
    n = y.size
    k = math.ceil(math.sqrt(n))
    n_new = k * k
    pad_len = n_new - n
    if pad_len:
        y = np.pad(y, (0, pad_len), mode='constant', constant_values=0)
    y = y.reshape(k, k)
    test_img = test_img.squeeze().numpy().reshape(32, 32)
    data = {'test_img': test_img, 'measurements': y, "forward_op": A}
    np.savez(save_dir / 'data.npz', **data)

    # save the test image
    plt.figure(figsize=(2,1))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_img, cmap='inferno')
    plt.savefig(save_dir / 'test_img.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # save the measurements
    plt.figure(figsize=(2,1))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y, cmap='inferno')
    plt.savefig(save_dir / 'measurements.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Test image and measurements are saved to {save_dir}!")


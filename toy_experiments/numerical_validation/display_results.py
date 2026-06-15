
import torch 
import math
from pathlib import Path
from typing import List, Dict
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_samples(X, mus, covs, prior_mus, prior_covs, pca, test_img, save_path):
    def plot_cov_ellipse(cov, mean, ax, n_std=1.0, **kwargs):
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = n_std * np.sqrt(eigvals) # this was 2
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellipse)

    # Set random seed
    np.random.seed(42)

    # Define parameters
    n_samples, dim = X.shape

    # Create two Gaussian modes
    mu1 = mus[0]; mu2 = mus[1]
    cov1 = covs[0]; cov2 = covs[1]
    prior_mu1 = prior_mus[0]; prior_mu2 = prior_mus[1]
    prior_cov1 = prior_covs[0]; prior_cov2 = prior_covs[1]

    print("test img.shape:", test_img.shape)
    test_img = pca.transform(test_img.reshape(1, -1))[0]

    # Apply PCA (to posterior samples)
    X_2d = pca.transform(X)
    mu1_2d = pca.transform(mu1.reshape(1, -1))[0]
    mu2_2d = pca.transform(mu2.reshape(1, -1))[0]
    P = pca.components_
    cov1_2d = P @ cov1 @ P.T
    cov2_2d = P @ cov2 @ P.T

    prior_mu1_2d = pca.transform(prior_mu1.reshape(1, -1))[0]
    prior_mu2_2d = pca.transform(prior_mu2.reshape(1, -1))[0]
    prior_cov1_2d = P @ prior_cov1 @ P.T
    prior_cov2_2d = P @ prior_cov2 @ P.T

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5, label='LMC Samples')
    ax.scatter(*mu1_2d, color='red', label='Male Mode')
    ax.scatter(*mu2_2d, color='blue', label='Female Mode')
    ax.scatter(*test_img, color='green', marker='x', label='Test Img', linewidth=3)
    # posterior ellipses
    plot_cov_ellipse(cov1_2d, mu1_2d, ax, edgecolor='red', facecolor='none', linewidth=2)
    plot_cov_ellipse(cov2_2d, mu2_2d, ax, edgecolor='blue', facecolor='none', linewidth=2)
    # prior ellipses
    plot_cov_ellipse(prior_cov1_2d, prior_mu1_2d, ax, edgecolor='red', facecolor='none', linestyle='--', alpha=0.5, linewidth=2)
    plot_cov_ellipse(prior_cov2_2d, prior_mu2_2d, ax, edgecolor='blue', facecolor='none', linestyle='--', alpha=0.5, linewidth=2)
    ax.legend()
    ax.set_title("PCA Projection of Two Gaussian Modes")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def save_test_img_y(test_img, y, save_path):
    """
    Visualized the test image and measurements in the same figure and save them to 
    save_path. 
    """

    fig, axes = plt.subplots(1, 2, figsize=(3, 2))

    # Plot image
    im1 = axes[0].imshow(test_img, cmap='inferno')
    axes[0].set_title('Test image\n (x)', fontsize=10)
    axes[0].axis('off')

    # Add horizontal colorbar under image
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("bottom", size="5%", pad=0.1)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')

    # Plot measurements
    num_pix = math.floor(math.sqrt(y.shape[0]))
    y_vis = y[:int(num_pix ** 2)].reshape((num_pix, num_pix))
    im2 = axes[1].imshow(y_vis, cmap='inferno')
    axes[1].set_title("Measurements\n" +r'($y = Ax + e$)', fontsize=10)
    axes[1].axis('off')

    # Add horizontal colorbar under measurements
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("bottom", size="5%", pad=0.1)
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Test image and measurements are saved to {str(save_path)}!")

def plot_save_stats(stats, img_shape=(32, 32), smoothing=False, fig_title='Statistical Maps', save_path=None, gamma=1):
    """
    Plot mean and variance images using Matplotlib.

    Parameters:
    - stats: dict containing 'mean_overall', 'var_overall', 'mus', 'covs' tensors
    - img_shape: shape to reshape the tensors to (H, W).
    - smoothing: whether to apply Gaussian smoothing.
    - dist_type: 'Prior' or 'Post.'
    - fig_title: main title of the figure.
    - save_path: if given, saves the figure as PNG to this path.
    """
    MEAN_CMAP = 'inferno'
    VAR_CMAP = 'viridis'

    fig, axes = plt.subplots(2, 3, figsize=(4, 3), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})

    num_modes = stats['mus'].shape[0]

    # get mean and variance images
    means = [stats['mean_overall'].cpu(),] + [stats['mus'][i].cpu() for i in range(num_modes-1,-1,-1)]
    varss = [stats['var_overall'].cpu(),] + [torch.diag(stats['covs'][i]).cpu() for i in range(num_modes-1,-1,-1)]

    # Gamma correction to make variance plots appear brighter
    if gamma != 1:
        # gamma < 1 brightens, > 1 darkens
        var_cmap = mcolors.ListedColormap(plt.cm.viridis(np.linspace(0, 1, 256))**[1, 1, 1, 1])  # copy
        var_norm = mcolors.PowerNorm(gamma=gamma)
    else: 
        var_norm = None

    for j in range(len(means)):
        mean_img = means[j].reshape(img_shape).cpu()
        var_img = varss[j].reshape(img_shape).cpu()

        if smoothing:
            mean_img = scipy.ndimage.gaussian_filter(mean_img, sigma=0.5)
            var_img = scipy.ndimage.gaussian_filter(var_img, sigma=0.5)

        im0 = axes[0, j].imshow(mean_img, cmap=MEAN_CMAP)
        txt = 'Overall' if j == 0 else ('Male' if j == 2 else 'Female')
        axes[0, j].set_title(txt, fontsize=10)
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
        axes[0, j].axis('off')

        # Mean colorbar below image
        cbar0 = plt.colorbar(im0, ax=axes[0, j], orientation='horizontal',
                    fraction=0.046, pad=0.03, shrink=0.5, aspect=10)
        cbar0.set_ticks([])
        pos1 = -0.35 if mean_img.min() < 0 else -0.3
        pos2 = 1.35 if mean_img.min() < 0 else 1.3
        cbar0.ax.text(pos1, 0.25, f"{mean_img.min():.1f}", va='center', ha='left', transform=cbar0.ax.transAxes, fontsize=5)
        cbar0.ax.text(pos2, 0.25, f"{mean_img.max():.1f}", va='center', ha='right', transform=cbar0.ax.transAxes, fontsize=5)

        max_var = torch.quantile(var_img, 0.99)  # or use a fixed value like 0.03
        var_img = torch.clamp(var_img, max=max_var)

        im1 = axes[1, j].imshow(var_img, cmap=VAR_CMAP, norm=var_norm)
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
        axes[1, j].axis('off')

        # Variance colorbar below image
        cbar1 = plt.colorbar(im1, ax=axes[1, j], orientation='horizontal',
                    fraction=0.046, pad=0.03, shrink=0.5, aspect=10)

        cbar1.set_ticks([])
        cbar1.ax.text(-0.37, 0.25, f"{var_img.min():.2f}", va='center', ha='left', transform=cbar1.ax.transAxes, fontsize=5)
        cbar1.ax.text(1.37, 0.25, f"{var_img.max():.2f}", va='center', ha='right', transform=cbar1.ax.transAxes, fontsize=5)

    #fig.suptitle(fig_title, fontsize=12)
    axes[0, 0].set_ylabel("Mean", fontsize=12, rotation=90, labelpad=6)
    axes[1, 0].set_ylabel("Variance", fontsize=12, rotation=90, labelpad=6)

    # Tighten layout
    plt.subplots_adjust(wspace=0.05, hspace=0.25, top=0.85)

    if save_path is None:
        print("Figure is being displayed!")
        plt.show()
    else:
        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def main(args):
    exp_no = [0, 1, 3, 5, 7]
    source_dir = Path('./results')

    source_dirs = list(source_dir.glob('*_effect'))
    for exp_dir in source_dirs:
        results_dirs = list(exp_dir.glob('*'))
        for idx, results_dir in enumerate(results_dirs):
            print(f"Experiment No: {idx + 1}/{len(results_dirs)}")

            kl_path = results_dir / 'kl_loss.npy'
            fi_path = results_dir / 'fi_loss.npy'

            if not kl_path.is_file():
                print("KL loss does not exist! Continue with the for loop...")
                continue

            if not fi_path.is_file():
                print("FI loss does not exist! Continue with the for loop...")
                continue

            # KL divergence calculation
            kl_loss = np.load(kl_path)
            kl_loss = torch.from_numpy(kl_loss)[exp_no]

            # FI calculation
            fi_loss = np.load(fi_path)
            fi_loss = torch.from_numpy(fi_loss)[exp_no]

            # save the plot for these losses
            dest_dir = Path(str(results_dir).replace('results', 'results_0_1_3_5_7'))
            dest_dir.mkdir(exist_ok=True, parents=True)
            save_losses(dest_dir, kl_loss, fi_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Numerical Validation")
    
    # Add arguments
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()

    main(args)
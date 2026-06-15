import numpy as np
import json
from pathlib import Path
import random
import argparse
from tqdm import tqdm
import torch
import math
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from estimators_potential import SGD, SGDm, SGDe, STORM, PAGE

from tqdm import tqdm
import torch
import math
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import sys; sys.path.append('/scratch/gilbreth/sahinm/LangevinMonteCarlo/StochasticInverse/toy_experiments/numerical_validation')
from estimators_potential import SGD, SGDm, SGDe, STORM, PAGE

class StochasticLMC_GMM:
    def __init__(
        self,
        estimator,
        save_dir=None,
        num_samples=5000,
        num_iter=5000,
        step_size=1e-2,
        grad_noise_std=0.0,
        device="cpu",
        seed=42,
        save_every=100,
        grid_limit=50,
        num_cells=300,
        kde_bandwidth=1.0,
        kde_chunk_size=20_000,
        gmm8_radius=8
    ):
        self.estimator = estimator
        self.num_samples = num_samples
        self.num_iter = num_iter
        self.step_size = step_size
        self.grid_limit = grid_limit
        self.num_cells = num_cells
        self.grad_noise_std = grad_noise_std
        self.device = device
        self.save_every = save_every
        self.seed = seed
        self.kde_bandwidth = kde_bandwidth
        self.kde_chunk_size = kde_chunk_size
        self.gmm8_radius = gmm8_radius

        self.rng = torch.Generator(device=device).manual_seed(seed)

        self.init_8_component_gmm()

        if save_dir is not None:
            
            self.save_config(
                save_dir / "config.json"
            )

    def save_config(self, save_path):
        """
        Save experiment configuration as JSON.
        """

        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        def convert(obj):
            """
            Convert non-JSON-serializable objects.
            """

            if isinstance(obj, Path):
                return str(obj)

            if isinstance(obj, torch.device):
                return str(obj)

            if isinstance(obj, torch.Generator):
                return "torch.Generator"

            if isinstance(obj, torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                return obj.detach().cpu().tolist()

            if isinstance(obj, dict):
                return {
                    k: convert(v)
                    for k, v in obj.items()
                }

            if isinstance(obj, (list, tuple)):
                return [
                    convert(v)
                    for v in obj
                ]

            # estimator object
            if hasattr(obj, "__dict__"):
                return {
                    "__class__": obj.__class__.__name__,
                    **{
                        k: convert(v)
                        for k, v in obj.__dict__.items()
                    }
                }

            return obj

        # Exclude large runtime objects
        exclude_keys = {
            "mus",
            "covs",
            "inv_covs",
            "rng",
        }

        config = {
            k: convert(v)
            for k, v in self.__dict__.items()
            if k not in exclude_keys
        }

        with open(save_path, "w") as f:
            json.dump(
                config,
                f,
                indent=4
            )

        print(f"Configuration saved to: {save_path}")

    def get_grad_count(self):
        """
        Returns cumulative number of potential-gradient evaluations used
        by the estimator.

        Assumes estimator stores either:
            estimator.grads_used
        as a list or scalar.
        """
        if hasattr(self.estimator, "grads_used"):
            grads = self.estimator.grads_used

            if isinstance(grads, list):
                if len(grads) == 0:
                    return 0
                return int(sum(grads))

            return int(grads)

        # fallback: assume one full-batch score evaluation per iteration
        return None
        
    def initialize_particles(self, init_type="far_gaussian"):
        if init_type == "standard":
            x = torch.randn(
                self.num_samples, 2,
                generator=self.rng,
                device=self.device
            ) * 5.0

        elif init_type == "far_gaussian":
            # Far from all modes
            center = torch.tensor([30.0, 30.0], device=self.device)
            x = center + torch.randn(
                self.num_samples, 2,
                generator=self.rng,
                device=self.device
            ) * 3.0

        elif init_type == "single_mode":
            # All particles near one component only
            center = self.mus[0]
            x = center + torch.randn(
                self.num_samples, 2,
                generator=self.rng,
                device=self.device
            ) * 0.5

        elif init_type == "wide_gaussian":
            # Very diffuse cloud
            x = torch.randn(
                self.num_samples, 2,
                generator=self.rng,
                device=self.device
            ) * 15.0

        elif init_type == "wrong_ring":
            # Ring larger than target ring
            radius = 25
            theta = 2 * math.pi * torch.rand(
                self.num_samples,
                generator=self.rng,
                device=self.device
            )
            x = torch.stack([
                radius * torch.cos(theta),
                radius * torch.sin(theta)
            ], dim=1)

            x = x + torch.randn_like(x) * 1.0

        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        return x
        
    def estimate_bandwidth(self, samples):
        """
        Silverman's rule in 2D.
        """
        N, D = samples.shape
        std = samples.std(dim=0).mean()
        h = 1.06 * std * (N ** (-1.0 / (D + 4)))
        return h.clamp(min=1e-3)

    def init_8_component_gmm(self):
        K = 8
        radius = self.gmm8_radius #10.0
        angles = torch.linspace(0, 2 * math.pi, K + 1)[:-1]

        mus = []
        for theta in angles:
            mus.append([
                radius * torch.cos(theta),
                radius * torch.sin(theta)
            ])

        self.mus = torch.tensor(mus, dtype=torch.float32, device=self.device)

        self.covs = torch.stack([
            torch.eye(2, device=self.device)
            for _ in range(K)
        ])

        self.inv_covs = torch.linalg.inv(self.covs)
        self.pis = torch.ones(K, device=self.device) / K

    def log_gmm(self, x, mus, covs, pis):
        N, D = x.shape
        inv_covs = torch.linalg.inv(covs)

        diff = x[:, None, :] - mus[None, :, :]

        mahal = torch.einsum(
            "nkd,kde,nke->nk",
            diff,
            inv_covs,
            diff
        )

        log_det = torch.logdet(covs)

        log_norm = -0.5 * (
            D * math.log(2 * math.pi) + log_det
        )

        log_probs = log_norm[None, :] - 0.5 * mahal
        log_joint = torch.log(pis[None, :] + 1e-12) + log_probs

        return torch.logsumexp(log_joint, dim=1)

    def score_gmm(self, x):
        N, D = x.shape

        diff = x[:, None, :] - self.mus[None, :, :]

        mahal = torch.einsum(
            "nkd,kde,nke->nk",
            diff,
            self.inv_covs,
            diff,
        )

        log_det = torch.logdet(self.covs)

        log_norm = -0.5 * (
            D * math.log(2 * math.pi) + log_det
        )

        log_probs = log_norm[None, :] - 0.5 * mahal

        log_joint = (
            torch.log(self.pis[None, :] + 1e-12)
            + log_probs
        )

        responsibilities = torch.softmax(log_joint, dim=1)

        component_scores = torch.einsum(
            "kde,nkd->nke",
            self.inv_covs,
            self.mus[None, :, :] - x[:, None, :]
        )

        score = torch.sum(
            responsibilities[:, :, None] * component_scores,
            dim=1
        )

        return score

    def sample_true_gmm(self):
        component_ids = torch.multinomial(
            self.pis,
            num_samples=self.num_samples,
            replacement=True,
            generator=self.rng
        )

        samples = []

        for k in component_ids:
            mvn = MultivariateNormal(
                loc=self.mus[k],
                covariance_matrix=self.covs[k]
            )
            samples.append(mvn.sample())

        return torch.stack(samples).cpu()

    def sample_langevin(self):
        x = self.initialize_particles(init_type=getattr(self, 'init_type', 'standard'))
        saved_grad_counts = [0]

        particles_history = []
        saved_iters = []

        particles_history.append(x.detach().cpu().clone())
        saved_iters.append(0)
        
        m_k_1 = None
        m_k_2 = None
        
        x_prev_1 = None
        x_prev_2 = None

        pbar = tqdm(range(1, self.num_iter + 1), desc="Running stochastic LMC")

        for k in pbar:
            score = self.score_gmm(x)
            
            score = self.estimator.score_update(
                x_k=x,
                x_k_1=x_prev_1,
                x_k_2=x_prev_2,
                m_k_1=m_k_1,
                m_k_2=m_k_2,
                score_fn=self.score_gmm,
                std_noise_grad=self.grad_noise_std,
                rng=self.rng,
            )

            x_new = (
                x
                + self.step_size * score
                + math.sqrt(2 * self.step_size) * torch.randn(x.shape, generator=self.rng, device=self.device, dtype=x.dtype)
            )

            x_prev_2 = x_prev_1
            x_prev_1 = x.detach().clone()

            m_k_2 = m_k_1
            m_k_1 = score.detach().clone()

            x = x_new
            
            if k % self.save_every == 0 or k == self.num_iter:
                particles_history.append(x.detach().cpu().clone())
                saved_iters.append(k)

                saved_grad_counts.append(
                    self.get_grad_count()
                )

            if k % 100 == 0:
                pbar.set_postfix({
                    "mean_norm": x.norm(dim=1).mean().item()
                })

        return (
            x.detach().cpu(),
            torch.stack(particles_history, dim=0),
            torch.tensor(saved_iters),
            torch.tensor(saved_grad_counts)
        )

    def kde_log_prob_and_score(self, grid, samples):
        """
        KDE estimate of log p(x) and score ∇ log p(x).

        grid:    [M, 2]
        samples: [N, 2]
        """
        if self.kde_bandwidth == "silverman":
            h = self.estimate_bandwidth(samples)
        else:
            h = float(self.kde_bandwidth)
        D = grid.shape[1]
        N = samples.shape[0]

        log_p_all = []
        score_all = []

        for start in range(0, grid.shape[0], self.kde_chunk_size):
            end = min(start + self.kde_chunk_size, grid.shape[0])
            grid_chunk = grid[start:end]

            diff = grid_chunk[:, None, :] - samples[None, :, :]  # [M_chunk, N, 2]
            sq_dist = torch.sum(diff ** 2, dim=-1)

            log_kernel = -0.5 * sq_dist / (h ** 2)

            log_p = torch.logsumexp(log_kernel, dim=1)
            log_p = log_p - math.log(N)
            log_p = log_p - D * math.log(h)
            log_p = log_p - 0.5 * D * math.log(2 * math.pi)

            weights = torch.softmax(log_kernel, dim=1)

            score = torch.sum(
                weights[:, :, None] * (-diff / (h ** 2)),
                dim=1
            )

            log_p_all.append(log_p)
            score_all.append(score)

        log_p_all = torch.cat(log_p_all, dim=0)
        score_all = torch.cat(score_all, dim=0)

        return log_p_all, score_all

    def calculate_losses(
        self, 
        particles_history,
        saved_iters,
        eps=1e-12, 
        fi_mode="mc"
    ):
        """
        KDE-based KL, Fisher Information, and TV.

        No GMM fitting is used here.
        """
        limit = self.grid_limit
        num_cells = self.num_cells

        x1 = torch.linspace(
            -limit,
            limit,
            num_cells,
            device=self.device
        ).float()

        x2 = torch.linspace(
            -limit,
            limit,
            num_cells,
            device=self.device
        ).float()

        x1, x2 = torch.meshgrid(x1, x2, indexing="ij")
        grid = torch.stack([x1.ravel(), x2.ravel()], dim=-1)

        # target GMM density and score
        log_q = self.log_gmm(
            grid,
            self.mus,
            self.covs,
            self.pis
        )

        q = torch.exp(log_q)
        q = q / (q.sum() + eps)
        log_q = torch.log(q + eps)

        score_q = self.score_gmm(grid)

        kl_list = []
        fi_list = []
        tv_list = []

        for idx in tqdm(
            range(particles_history.shape[0]),
            desc="Calculating KDE KL/FI/TV losses"
        ):
            particles = particles_history[idx].to(self.device)

            log_p, score_p = self.kde_log_prob_and_score(
                grid,
                particles
            )

            p = torch.exp(log_p)
            p = p / (p.sum() + eps)
            log_p = torch.log(p + eps)

            # KL(p || q)
            kl = torch.sum(p * (log_p - log_q))

            # FI(p || q)
            # Monte Carlo FI, more stable than grid FI
            if fi_mode == "mc":
                _, score_p_particles = self.kde_log_prob_and_score(
                    particles,
                    particles
                )

                score_q_particles = self.score_gmm(particles)

                fi = torch.mean(
                    torch.sum(
                        (score_p_particles - score_q_particles) ** 2,
                        dim=-1
                    )
                )
            else:
                fi = torch.sum(
                    p * torch.sum(
                        (score_p - score_q) ** 2,
                        dim=-1
                    )
                )

            # TV(p, q)
            tv = 0.5 * torch.sum(torch.abs(p - q))

            kl_list.append(kl.detach().cpu().item())
            fi_list.append(fi.detach().cpu().item())
            tv_list.append(tv.detach().cpu().item())

        return {
            "iter": saved_iters.cpu(),
            "KL": torch.tensor(kl_list),
            "FI": torch.tensor(fi_list),
            "TV": torch.tensor(tv_list),
        }

    def compare_plots(
        self,
        particles_history,
        saved_iters,
        saved_grad_counts=None,
        checkpoints=(0, 50, 100, 250, 2000),
        plot_limit=20,
        save_path=None,
        x_axis="iter",
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.1,
        })

        if x_axis == "iter":
            axis_values = saved_iters.cpu().numpy()
            title_prefix = "Iter"
        elif x_axis == "grad":
            if saved_grad_counts is None:
                raise ValueError("saved_grad_counts must be provided when x_axis='grad'")
            axis_values = saved_grad_counts.cpu().numpy()
            title_prefix = "Grad"
        else:
            raise ValueError("x_axis must be either 'iter' or 'grad'")

        selected_idx = []
        selected_values = []

        for ckpt in checkpoints:
            idx = np.argmin(np.abs(axis_values - ckpt))
            selected_idx.append(idx)
            selected_values.append(axis_values[idx])

        fig, axes = plt.subplots(
            1,
            len(checkpoints) + 1,
            figsize=(3.2 * (len(checkpoints) + 1), 3.4),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        for ax, idx, axis_value in zip(axes[:-1], selected_idx, selected_values):
            particles = particles_history[idx]

            ax.scatter(
                particles[:, 0],
                particles[:, 1],
                s=1.2,
                alpha=0.45,
                rasterized=True,
            )

            ax.scatter(
                self.mus[:, 0].cpu(),
                self.mus[:, 1].cpu(),
                marker="x",
                s=55,
                linewidths=1.4,
            )

            ax.set_title(f"{title_prefix} {int(axis_value)}")

        true_samples = self.sample_true_gmm()

        axes[-1].scatter(
            true_samples[:, 0],
            true_samples[:, 1],
            s=1.2,
            alpha=0.45,
            rasterized=True,
        )

        axes[-1].scatter(
            self.mus[:, 0].cpu(),
            self.mus[:, 1].cpu(),
            marker="x",
            s=55,
            linewidths=1.4,
        )

        axes[-1].set_title("Ground truth")

        for ax in axes:
            ax.set_xlim(-plot_limit, plot_limit)
            ax.set_ylim(-plot_limit, plot_limit)
            ax.set_aspect("equal")

            ax.grid(
                True,
                linestyle="--",
                linewidth=0.5,
                alpha=0.35,
            )

            ax.tick_params(
                direction="in",
                length=3,
                width=0.8,
            )

            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

        axes[0].set_ylabel(r"$x_2$")

        for ax in axes:
            ax.set_xlabel(r"$x_1$")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

        return true_samples

def plot_losses(
    results, 
    save_path=None, 
    tail_window=100, 
    show=True,
    x_axis='iter'
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from pathlib import Path

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
    })

    eps = 1e-12

    iters = results["iter"].cpu().numpy()

    if x_axis == "iter":
        x = results["iter"].cpu().numpy()
        xlabel = "Iteration"
    elif x_axis == "grad":
        x = results["grad_counts"].cpu().numpy()
        xlabel = "# of gradients per sample"
    else:
        raise ValueError("x_axis must be 'iter' or 'grad'")

    kl = results["KL"].cpu().numpy() + eps
    fi = results["FI"].cpu().numpy() + eps
    tv = results["TV"].cpu().numpy() + eps

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    metrics = [
        (kl, "KL divergence"),
        (fi, "Fisher information"),
        (tv, "Total variation"),
    ]

    for ax, (vals, ylabel) in zip(axes, metrics):
        ax.plot(x, vals, linewidth=2.0)
        ax.set_xlabel(xlabel)

        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        
        n_tail = min(tail_window, len(vals))
        conv_val = vals[-n_tail:].mean()
        
        ax.axhline(
            conv_val,
            linestyle="--",
            linewidth=1.4,
            alpha=0.7,
        )

        ax.annotate(
            f"{conv_val:.4f}",
            xy=(0.96, conv_val),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0, 8),   # move up by 8 points
            textcoords="offset points",
            fontsize=12,
            ha="right",
            va="bottom",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=1.5,
            ),
        )

        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.02)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_estimator(args):
    """
    Create a fresh estimator object for each independent run.
    This avoids carrying estimator state, e.g., grads_used or momentum,
    across different initializations.
    """
    match args.estimator:
        case 'sgdm':
            return SGDm(beta=args.beta, mini_batch_size=args.batch_size)
        case 'sgde':
            return SGDe(beta=args.beta, batch_size=args.batch_size)
        case 'storm':
            return STORM(beta=args.beta, mini_batch_size=args.batch_size)
        case 'page':
            return PAGE(
                beta=args.beta,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
            )
        case _:
            print('default estimator SGD is chosen!')
            return SGD(mini_batch_size=args.batch_size)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_aggregate_losses(
    aggregate_results,
    save_path=None,
    x_axis='iter',
):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    eps = 1e-12

    if x_axis == 'iter':
        x = aggregate_results['iter']
        xlabel = 'Iteration'
    elif x_axis == 'grad':
        x = aggregate_results['saved_grad_counts']
        xlabel = '# of gradients per sample'
    else:
        raise ValueError("x_axis must be 'iter' or 'grad'")

    metrics = [
        ('kl', 'KL divergence'),
        ('fi', 'Fisher information'),
        ('tv', 'Total variation'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax, (key, ylabel) in zip(axes, metrics):
        mean = aggregate_results[f'{key}'] + eps
        std = aggregate_results[f'{key}_std']

        lower = np.maximum(mean - std, eps)
        upper = mean + std

        ax.plot(x, mean, linewidth=2.0, label='mean')
        ax.fill_between(x, lower, upper, alpha=0.20, label=r'$\pm$ 1 std')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.45)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(x=0.02)
        ax.legend(frameon=False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Langevin Monte Carlo sampling script.")

    # experiment parameters
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to simulate LMC')
    parser.add_argument('--num_iter', type=int, default=2000, help='Number of iterations to run LMC')
    parser.add_argument('--num_cells', type=int, default=100, help='Number of cells to discretize the grid')
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'), help='device to run the experiments')
    parser.add_argument('--grid_limit', type=int, default=50, help='Grid limit for calculating the loss')
    parser.add_argument('--save_dir', type=str, default='./results', help='directory path for saving results')
    parser.add_argument('--save_every', type=int, default=20, help='period saving particles')
    parser.add_argument('--gmm8_radius', type=float, default=30.0, help='radius of the GMMs')
    parser.add_argument('--num_runs', type=int, default=1, help='number of independent initializations')
    parser.add_argument('--base_seed', type=int, default=42, help='base random seed; run r uses base_seed + r')
    parser.add_argument('--init_type', type=str, default='standard', choices=('standard', 'far_gaussian', 'single_mode', 'wide_gaussian', 'wrong_ring'), help='particle initialization type')
    parser.add_argument('--save_particles', action='store_true', help='save particle histories for every run')

    # LMC algorithm parameters
    parser.add_argument('--std_noise_grad', type=float, default=50.0, help='Standard deviation of the SFO')
    parser.add_argument('--gamma', type=float, default=1e-2, help='Step size for LMC algorithm')
    parser.add_argument('--batch_size', type=int, default=10, help='large batch size for PAGE-type algorithms')
    parser.add_argument('--mini_batch_size', type=int, default=2, help='mini-batch size for PAGE correction step')
    parser.add_argument('--estimator', type=str, default='sgd', choices=('sgd', 'sgde', 'sgdm', 'page', 'storm', 'eve', 'zo_sgd'), help='estimator to choose for score calculation')
    parser.add_argument('--beta', type=float, default=0.99, help='momentum/probability parameter')

    # Display settings
    parser.add_argument('--tail_window', type=int, default=5, help='tail to take average')
    parser.add_argument('--make_run_plots', action='store_true', help='save scatter/loss plots for each individual run')

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_kl = []
    all_fi = []
    all_tv = []
    all_grad_counts = []
    saved_iters_ref = None
    mus_ref = None

    for run_id in range(args.num_runs):
        seed = args.base_seed + run_id
        set_all_seeds(seed)

        run_dir = save_dir / f'run_{run_id:02d}_seed_{seed}'
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n========== Run {run_id + 1}/{args.num_runs} | seed={seed} ==========', flush=True)

        estimator = make_estimator(args)

        sampler = StochasticLMC_GMM(
            save_dir=run_dir,
            estimator=estimator,
            num_samples=args.num_samples,
            num_iter=args.num_iter,
            step_size=args.gamma,
            grad_noise_std=args.std_noise_grad,
            num_cells=args.num_cells,
            kde_bandwidth='silverman',
            save_every=args.save_every,
            grid_limit=args.grid_limit,
            gmm8_radius=args.gmm8_radius,
            device=args.device,
            seed=seed,
        )

        # This line lets you choose different initializations via --init_type.
        # The sampler method below uses this attribute if present.
        sampler.init_type = args.init_type

        true_samples = sampler.sample_true_gmm()
        langevin_samples, particles_history, saved_iters, saved_grad_counts = sampler.sample_langevin()

        print('Experiment completed!')
        print('Losses are being calculated...')
        results = sampler.calculate_losses(particles_history, saved_iters)
        results['grad_counts'] = saved_grad_counts
        print('Losses are calculated!')

        all_kl.append(tensor_to_numpy(results['KL']))
        all_fi.append(tensor_to_numpy(results['FI']))
        all_tv.append(tensor_to_numpy(results['TV']))
        all_grad_counts.append(tensor_to_numpy(saved_grad_counts))

        if saved_iters_ref is None:
            saved_iters_ref = tensor_to_numpy(saved_iters)
            mus_ref = sampler.mus.detach().cpu().numpy()

        run_data = {
            'seed': seed,
            'fi': tensor_to_numpy(results['FI']),
            'kl': tensor_to_numpy(results['KL']),
            'tv': tensor_to_numpy(results['TV']),
            'true_samples': tensor_to_numpy(true_samples),
            'saved_iters': tensor_to_numpy(saved_iters),
            'saved_grad_counts': tensor_to_numpy(saved_grad_counts),
            'final_samples': tensor_to_numpy(langevin_samples),
            'mus': mus_ref,
        }

        if args.save_particles:
            run_data['particle_history'] = tensor_to_numpy(particles_history)

        np.savez(run_dir / 'results.npz', **run_data)

        if args.make_run_plots:
            true_samples = sampler.compare_plots(
                particles_history,
                saved_iters,
                saved_grad_counts=saved_grad_counts,
                checkpoints=(0, 100, 200, 300, args.num_iter),
                plot_limit=40,
                save_path=run_dir / 'scatter_plots_vs_iter.png',
                x_axis='iter',
            )

            _ = sampler.compare_plots(
                particles_history,
                saved_iters,
                saved_grad_counts=saved_grad_counts,
                checkpoints=(0, 101, 201, 301, int(saved_grad_counts[-1].item())),
                plot_limit=40,
                save_path=run_dir / 'scatter_plots_vs_grads.png',
                x_axis='grad',
            )

            plot_losses(
                results,
                save_path=run_dir / 'losses_vs_iter.png',
                tail_window=args.tail_window,
                show=False,
                x_axis='iter',
            )

            plot_losses(
                results,
                save_path=run_dir / 'losses_vs_grads.png',
                tail_window=args.tail_window,
                show=False,
                x_axis='grad',
            )

        del sampler, estimator, particles_history, langevin_samples, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_kl = np.stack(all_kl, axis=0)
    all_fi = np.stack(all_fi, axis=0)
    all_tv = np.stack(all_tv, axis=0)
    all_grad_counts = np.stack(all_grad_counts, axis=0)

    aggregate_results = {
        'iter': saved_iters_ref,
        'kl_all': all_kl,
        'fi_all': all_fi,
        'tv_all': all_tv,
        'saved_grad_counts_all': all_grad_counts,
        'kl': all_kl.mean(axis=0),
        'kl_std': all_kl.std(axis=0),
        'fi': all_fi.mean(axis=0),
        'fi_std': all_fi.std(axis=0),
        'tv': all_tv.mean(axis=0),
        'tv_std': all_tv.std(axis=0),
        'saved_grad_counts': all_grad_counts.mean(axis=0),
        'saved_grad_counts_std': all_grad_counts.std(axis=0),
        'seeds': np.arange(args.base_seed, args.base_seed + args.num_runs),
        'mus': mus_ref,
    }

    np.savez(save_dir / 'aggregate_results.npz', **aggregate_results)

    plot_aggregate_losses(
        aggregate_results,
        save_path=save_dir / 'aggregate_losses_vs_iter.png',
        x_axis='iter',
    )

    plot_aggregate_losses(
        aggregate_results,
        save_path=save_dir / 'aggregate_losses_vs_grads.png',
        x_axis='grad',
    )

    with open(save_dir / 'aggregate_summary.json', 'w') as f:
        json.dump(
            {
                'num_runs': args.num_runs,
                'base_seed': args.base_seed,
                'seeds': list(range(args.base_seed, args.base_seed + args.num_runs)),
                'final_KL_mean': float(aggregate_results['kl'][-1]),
                'final_KL_std': float(aggregate_results['kl_std'][-1]),
                'final_FI_mean': float(aggregate_results['fi'][-1]),
                'final_FI_std': float(aggregate_results['fi_std'][-1]),
                'final_TV_mean': float(aggregate_results['tv'][-1]),
                'final_TV_std': float(aggregate_results['tv_std'][-1]),
            },
            f,
            indent=4,
        )

    print(f'\nAll runs completed. Results saved to: {save_dir}')
    print(f'Aggregate results saved to: {save_dir / "aggregate_results.npz"}')


if __name__ == '__main__':
    main()

import numpy as np
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
#from scipy.stats import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
import copy 
from typing import Tuple, Optional, List
#from sklearn.mixture import GaussianMixture
import torch
import random
from gmm import GaussianMixture
import matplotlib.pyplot as plt 
from utils import generate_informative_A, generate_acceptable_A, bounded_noise
from estimators import SGD, SGDm, SGDe, STORM, PAGE, EVE, ZO_SGD
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

class LangevinMonteCarlo:
    
    """
    This class is designed for the numerical validation of Langevin Monte Carlo for 
    posterior sampling. 
    """
    
    def __init__(
        self, 
        estimator: object,
        num_exps: int = 1,
        num_iter: int = 1000,
        num_samples: int = 1000,
        num_cells: int = 1000, 
        grid_limit: float = 50,
        xi: float = 0.975, 
        sigma0: float = 10.0, 
        alpha0: float = 10.0, 
        sigmin: float = 0.0, 
        epsmax: float = 2.5,
        std_noise_grad: float = 0.0,
        gamma: float = 0.8,
        y_std: float = 1.0, 
        device : str = 'cpu',
        seed: int = 42,
        noise_type: str = 'gauss',
        save_dir: str | Path = './results'
    ):
        assert device in ('cuda', 'cpu')
        assert noise_type in ('laplace', 'gauss'), "Available noise types are 'laplace' or 'gauss'"

        # experiment parameters 
        self.num_exps = num_exps
        self.num_iter = num_iter 
        self.num_samples = num_samples
        self.num_cells = num_cells
        self.grid_limit = grid_limit
        self.device = device
        self.save_dir = save_dir

        # annealing parameters
        self.sigmin = sigmin
        self.sigma0 = sigma0
        self.alpha0 = alpha0
        self.epsmax = epsmax
        self.xi = xi 
        
        # inverse problem parameters 
        self.y_std = y_std
        self.noise_type = noise_type
        
        # LMC algorithm parameters
        self.std_noise_grad = std_noise_grad
        self.gamma = gamma 
        self.gamma_init = gamma
        
        # other parameters
        self.rng = torch.Generator(device).manual_seed(seed)
        self.seed = seed

        self.estimator_dict = {'estimator_name': estimator.__class__.__name__}
        self.estimator_dict.update(estimator.__dict__)

        # save the parameters to the save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_config(save_dir / 'configs.json')

        self.estimator = estimator

        if hasattr(estimator, "batch_size"):
            self.batch_size_init = estimator.batch_size

        # get sigma and alpha schedules
        self.sigmas, self.alphas = self.get_sigma_alpha_schedule()
        
        # initialize Gaussian mixture for prior
        self.init_variables()

    def save_config(self, path: str):
        def convert(o):
            # Convert non-serializable types
            if isinstance(o, torch.device):
                return str(o)
            if isinstance(o, torch.Generator):
                return "torch.Generator"  # or None or str(o)
            if isinstance(o, dict):
                return {k: convert(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [convert(v) for v in o]
            return o

        config = {k: convert(v) for k, v in self.__dict__.items()}
        
        # Only include simple data, skip things like tensors or other objects
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
        
    def get_sigma_alpha_schedule(self):
        sigmas = torch.max(self.sigma0 *  self.xi ** torch.arange(self.num_iter), torch.tensor(self.sigmin))
        alphas = torch.max(self.alpha0 * sigmas ** 2, torch.tensor(1.0))
        return sigmas.to(self.device), alphas.to(self.device)

    def score_likelihood(self, y: torch.Tensor, x: torch.Tensor, A: torch.Tensor):
        '''calculates the score function of the likelihood (y = Ax + e) for data points, x'''
        tmp = -(y - torch.einsum('xz,z->x', x, A)) / self.y_std ** 2
        return -tmp[..., None] * A # -1 is removed because this outputs the grad of potential function of likelihood, i.e., V in exp(-V)
  
    def init_variables(self):
        # initialize prior parameters
        self.mus = torch.tensor([[-20, -20], [20, 20]], dtype=torch.float32, device=self.device)
        #self.mus = torch.tensor([[-10, -10], [10, 10]], dtype=torch.float32, device=self.device)
        self.covs = torch.tensor([
            [[10, 4], [4, 10]], 
            [[12, -6],[-6, 12]]
        ], dtype=torch.float32, device=self.device)
        self.pis = torch.tensor([0.5, 0.5], dtype=torch.float32, device=self.device)
        self.inv_covs = torch.linalg.inv(self.covs)

    def log_gmm(self,
        x: torch.Tensor,
        mus: torch.Tensor, 
        covs: torch.Tensor, 
        pis: torch.Tensor
    )->torch.Tensor:
        """calculates the log-prob of the GMM prior""" 
        # calculate log-prior
        K = mus.shape[0]
        log_probs = []
        for k in range(K):
            mvns = MultivariateNormal(loc=mus[k], covariance_matrix=covs[k])
            tmp = mvns.log_prob(x) # (N,)
            log_probs.append(tmp)
        log_probs = torch.stack(log_probs, dim=0)
        log_pis = torch.log(pis + 1e-12).unsqueeze(dim=1) # (K, 1)
        log_probs = torch.logsumexp(log_pis + log_probs, dim=0)
        return log_probs

    def log_likelihood(self, 
        y: torch.Tensor, 
        x: torch.Tensor, 
        A: torch.Tensor
    )->torch.Tensor:

        """calculates the log-likelihood of the forward model with Gaussian noise"""

        assert y.device == x.device == A.device, "all variables must be on the same device!"

        if self.noise_type == 'gauss':
            # calculate the log-likelihood for each gaussian likelihood
            y = y.unsqueeze(0)              # [1, D]
            means = torch.einsum('xy,y->x', x, A).unsqueeze(1)              # [N, 1]
            var = torch.ones(means.shape[0], 1, device=y.device) * self.y_std ** 2

            log_2pi = torch.log(torch.tensor(2 * torch.pi)).unsqueeze(0).unsqueeze(1).to(y.device)
            log_var = torch.log(var)
            quad_term = (y - means) ** 2 / var
            # calculate log-probability
            log_prob = -0.5 * (log_2pi + log_var + quad_term).squeeze(1)
        else: # Laplacian noise
            y = y.unsqueeze(0)
            means = x @ A
            log_term = torch.log(torch.tensor(math.sqrt(2) * self.y_std))
            laplace_term = math.sqrt(2) / self.y_std * torch.abs(means)
            # calculate log-probability
            log_prob = - log_term - laplace_term

        return log_prob

    def score_gmm(
        self, 
        x: torch.Tensor, 
        mus: torch.Tensor, 
        covs: torch.Tensor, 
        pis: torch.Tensor,
        inv_covs: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        '''calculates the score function of the GMM prior for data points, x'''
        N, D = x.shape
        K    = pis.shape[0]
        device = x.device
        dtype  = x.dtype

        # ---------- log N_k(x) for every (sample, component) ----------
        diff = x.unsqueeze(1) - mus.unsqueeze(0)                     # (N, K, D)

        # Mahalanobis distance  (x-μ)ᵀ Σ⁻¹ (x-μ)
        if inv_covs is None:
            inv_covs = torch.linalg.inv(covs)
        m_dist2 = torch.einsum('nkd,kde,nke->nk', diff, inv_covs, diff)  # (N, K)

        # log |Σ_k|
        log_det = torch.logdet(covs)                                  # (K,)

        LOG2PI = math.log(2.0 * math.pi)
        log_norm = -0.5 * (log_det + D * LOG2PI)                      # (K,)
        log_pdf  = log_norm.unsqueeze(0) - 0.5 * m_dist2              # (N, K)

        # ---------- mixture weights in log-space ----------
        log_pi     = pis.log().unsqueeze(0)                           # (1, K)
        log_joint  = log_pi + log_pdf                                 # (N, K)
        log_weight = log_joint - torch.logsumexp(log_joint,
                                                dim=1, keepdim=True) # (N, K) responsibilities in log-space
        weight = log_weight.exp()                                     # (N, K)

        # ---------- component scores  Σ⁻¹(μ − x) ----------
        comp_scores = torch.einsum('kde,nkd->nke', inv_covs,
                                mus - x.unsqueeze(1))              # (N, K, D)

        # mixture score  ∑_k w_k Σ_xk⁻¹(μ_k − x)
        score = (weight.unsqueeze(2) * comp_scores).sum(dim=1)        # (N, D)
        return score

    def run_apmc_red(self, apply_scheduler: bool=False):
        # initialize the samples
        x_init = torch.rand(
            size=(self.num_samples, 2),
            device=self.device, 
            generator=self.rng,
        ) * 100 - 50 # sample from U[-50, 50]

        # initialize the measurement vector for x=(0,0)
        y = torch.normal(
            mean=0.0, 
            std=self.y_std, 
            size=(1,),
            generator=self.rng,
            device=self.device,
        )

        # initialize forward models
        forward_ops = [generate_acceptable_A(self.mus, self.covs, self.rng) for _ in range(self.num_exps)]
        forward_ops = torch.stack(forward_ops, dim=0)

        # generate multiple seeds
        seeds = torch.randint(0, 2**63 - 1, (4,), device=self.rng.device, generator=self.rng)
        # create new generators from these seeds
        [gen_eps, gen_sig, gen_ll, gen_z] = [torch.Generator(self.device).manual_seed(int(seed)) for seed in seeds]

        all_exps = []
        all_exps = {'particles': [], 'grads': []}
        for exp_no in range(self.num_exps):
            print(f"Experiment No: {exp_no+1}")
            # reset the gradients to remove the all grads from previous experiments
            self.estimator.reset_grads()
            # initialize particles
            x_k = x_init.clone()
            A = forward_ops[exp_no]
            # initialize the previous and current estimate of the score likelihood
            score_ll_est, score_ll_est_1, score_ll_est_2 = None, None, None
            # list for saving the particles for loss calculation later
            all_particles = [x_k.cpu().clone(),]
            for k in tqdm(range(self.num_iter), desc=f"γ={self.gamma}, σ_min={self.sigmin}, ϵ_max={self.epsmax}"):
                
                # apply scheduler for weak convergence
                if apply_scheduler:
                    self.scheduler(k)

                # calculate noisy prior score function
                noise_eps_k = bounded_noise(x_k.shape, gen_eps, self.device, self.epsmax) if self.epsmax != 0 else 0.0
                covs =  self.covs + self.sigmas[k] ** 2 * torch.stack([torch.eye(self.mus.shape[-1], device=self.mus.device) for _ in range(self.mus.shape[0])])
                inv_covs = torch.linalg.inv(covs).to(self.device) # calculate the precision matrix
                score_k = self.score_gmm(x_k, self.mus, covs, pis=self.pis, inv_covs=inv_covs) # calculate the analytical score of prior
                noisy_score_k = score_k + noise_eps_k # + noise_sig_k

                # calculate the estimate of the score likelihood
                x_k_1 = all_particles[-2] if k != 0 else None # return the particles at time t-1
                x_k_2 = all_particles[-3] if k > 1 else None # return the particles at time t-2

                score_ll_est = self.estimator.likelihood_update(
                    x_k=x_k, x_k_1=x_k_1, x_k_2=x_k_2,
                    y=y, A=A,
                    m_k_1=score_ll_est_1, m_k_2=score_ll_est_2,
                    std_noise_grad=self.std_noise_grad,
                    y_std=self.y_std,
                    rng=gen_ll, noise_type=self.noise_type
                )

                score_ll_est_2, score_ll_est_1 = score_ll_est_1, score_ll_est

                # estimate of the score of the posterior
                score_post_est = score_ll_est - self.alphas[k] * noisy_score_k
                
                # do the ALMC update
                z_k = torch.randn(size=x_k.shape, generator=gen_z, device=self.device)
                x_k = x_k - self.gamma * score_post_est + math.sqrt(2 * self.gamma) * z_k

                # save the particles at the current iterate
                all_particles.append(x_k.cpu().clone())

            # save all particles 
            all_particles = torch.stack(all_particles, dim=0)
            all_exps['particles'].append(all_particles)
            all_exps['grads'].append(torch.tensor(self.estimator.grads_used))
        
        # save the particles & forward operators of all experiments
        all_exps['particles'] = torch.stack(all_exps['particles'], dim=0)
        all_exps['grads'] = torch.stack(all_exps['grads'], dim=0)
        return all_exps, forward_ops.cpu(), y.cpu()
         
    def calculate_losses(self, 
        all_exps: torch.Tensor,
        forward_ops: torch.Tensor,
        y: torch.Tensor
    ):
        limit = self.grid_limit
        num_cells = self.num_cells

        # initialize grid for approximate KL, FI, and TV calculations
        x1 = torch.linspace(-limit + 0.5, limit - 0.5, num_cells, device=self.device).to(torch.float32)
        x2 = torch.linspace(-limit + 0.5, limit - 0.5, num_cells, device=self.device).to(torch.float32)
        x1, x2 = torch.meshgrid(x1, x2, indexing='ij')
        grid = torch.stack([x1.ravel(), x2.ravel()], dim=-1).to(self.device)
        
        pred_gmm_params = []
        kl_all, fi_all, tv_all = [], [], []
        for exp_idx in range(self.num_exps):
            A = forward_ops[exp_idx]

            # get the log q(x|y) and gradient log(q(x|y)) of the posterior over the grid
            y, A = y.to(self.device), A.to(self.device)
            # calculate the target log-prob. and prob. over the grid
            log_q = self.log_gmm(grid, self.mus, self.covs, self.pis) + self.log_likelihood(y, grid, A)
            log_q -= torch.logsumexp(log_q, dim=0)
            q = torch.exp(log_q)
            score_q = self.score_gmm(grid, self.mus, self.covs, self.pis, self.inv_covs) + self.score_likelihood(y, grid, A)

            # get the particles history of the experiment
            all_particles = all_exps[exp_idx]
            # lists for KL and FI losses 
            kl_list, fi_list, tv_list = [], [], []
            for k in tqdm(range(self.num_iter), desc=f'Calculating the loss for exp. {exp_idx+1}'):
                particles = all_particles[k].to(self.device)
                K, D, _ = self.covs.shape

                # fit Gaussian mixture to particles in substeps
                gmm = GaussianMixture(n_components=K, n_features=D, covariance_type='full', seed=self.seed)
                gmm.to(self.device)
                gmm.fit(particles)
                
                log_p = gmm.score_samples(grid)
                log_p -= torch.logsumexp(log_p, dim=0)
                p = torch.exp(log_p)

                mus = gmm.mu.clone().squeeze(0)
                covs = gmm.var.clone().squeeze(0)
                pis = gmm.pi.clone().squeeze(0).squeeze(1)
                score_p = self.score_gmm(grid, mus, covs, pis)

                # save the gmm in last iteration
                if k == self.num_iter-1:
                    pred_gmm_params.append({'mus': mus.cpu(), 'covs': covs.cpu(), 'pis': pis.cpu()})
                
                # calculate the approximate KL divergence over the grid
                kl_pointwise = p * (log_p - log_q)
                kl_total = torch.sum(kl_pointwise)# * cell_area
  
                # calculate the approximate FI over the grid 
                fi_pointwise = p * torch.sum((score_p - score_q) ** 2, dim=-1)
                fi_total = torch.sum(fi_pointwise)# * cell_area 

                # calculate the approximate TV distance over the grid
                tv_pointwise = torch.abs(p - q)
                tv_total = 0.5 * torch.sum(tv_pointwise)

                # save the loss values
                kl_total = kl_total.cpu().item()
                fi_total = fi_total.cpu().item()
                tv_total = tv_total.cpu().item()
                
                kl_list.append(kl_total)
                fi_list.append(fi_total)
                tv_list.append(tv_total)

            # save the experiment losses 
            kl_all.append(torch.tensor(kl_list))
            fi_all.append(torch.tensor(fi_list))
            tv_all.append(torch.tensor(tv_list))
            
        return {
            'iter': torch.tensor(list(range(0, self.num_iter,))),
            'KL'  : kl_all, 
            'FI'  : fi_all, 
            'TV'  : tv_all,
            'pred_gmm': pred_gmm_params
        }

    def save_losses(
        self,
        kl_results: torch.Tensor,
        fi_results: torch.Tensor,
        tv_results: torch.Tensor
    ):

        EPS = 0#1e-10

        ######################## GENERATE KL DIVERGENCE PLOT ########################
        mean = kl_results.mean(dim=0).numpy()
        kl_min = kl_results.min(dim=0)[0].numpy()
        kl_max = kl_results.max(dim=0)[0].numpy()
        #std = kl_results.std(dim=0).numpy() + EPS
        
        x = np.arange(len(mean))

        plt.figure(figsize=(6,4))
        plt.plot(x, mean, color='green')
        plt.fill_between(x, kl_min, kl_max, color='green', alpha=.2)

        # dashed line at minimum of the mean curve 
        #min_val = mean.min()
        min_val = mean[-min(len(mean), 100):].mean() 
        plt.axhline(min_val, linestyle='--', color='green', linewidth=1)
        label_offset = 0.3 * min_val
        plt.text(len(mean), min_val - label_offset, f'{min_val:.4f}', va='top', ha='right', fontsize=10)
        plt.ylim(1e-3, None)  # only set lower limit
        
        # Style
        plt.yscale('log')
        # Set major ticks only (e.g. 10^2, 10^1, 10^0, 10^-1, etc.)
        plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        # Disable minor ticks
        plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
        plt.xlabel('Iteration')
        plt.ylabel('KL\ndivergence')
        #plt.title(r'$\gamma = 1.6$')
        plt.tight_layout()

        # create the save directory
        save_dir = Path(self.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / 'kl_loss.png')
        plt.close()

        ######################## GENERATE FI PLOT ########################
        mean = fi_results.mean(dim=0).numpy()
        fi_min = fi_results.min(dim=0)[0].numpy()
        fi_max = fi_results.max(dim=0)[0].numpy()
        #std = fi_results.std(dim=0).numpy() + EPS
        x = np.arange(len(mean))

        plt.figure(figsize=(6,4))
        plt.plot(x, mean, color='purple')
        plt.fill_between(x, fi_min, fi_max, color='purple', alpha=.2)

        # dashed line at minimum of the mean curve 
        #min_val = mean.min()
        min_val = mean[-min(len(mean), 100):].mean() 
        plt.axhline(min_val, linestyle='--', color='purple', linewidth=1)
        label_offset = 0.4 * min_val
        plt.text(len(mean), min_val-label_offset, f'{min_val:.4f}', va='top', ha='right', fontsize=10)
        plt.ylim(1e-3, None)  # only set lower limit
        
        # Style
        plt.yscale('log')
        # Set major ticks only (e.g. 10^2, 10^1, 10^0, 10^-1, etc.)
        plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        # Disable minor ticks
        plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
        plt.xlabel('Iteration')
        plt.ylabel('Fisher\nInformation')
        plt.tight_layout()

        # create the save directory
        save_dir = Path(self.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / 'fi_loss.png')
        plt.close()

        ######################## GENERATE TV PLOT ########################
        mean = tv_results.mean(dim=0).numpy()
        tv_min = tv_results.min(dim=0)[0].numpy()
        tv_max = tv_results.max(dim=0)[0].numpy()
        #std = fi_results.std(dim=0).numpy() + EPS
        x = np.arange(len(mean))

        plt.figure(figsize=(6,4))
        plt.plot(x, mean, color='blue')
        plt.fill_between(x, tv_min, tv_max, color='blue', alpha=.2)

        # dashed line at minimum of the mean curve 
        #min_val = mean.min()
        min_val = mean[-min(len(mean), 100):].mean() 
        plt.axhline(min_val, linestyle='--', color='blue', linewidth=1)
        label_offset = 0.1 * min_val
        plt.text(len(mean), min_val-label_offset, f'{min_val:.4f}', va='top', ha='right', fontsize=10)
        plt.ylim(1e-3, None)  # only set lower limit
        
        # Style
        plt.yscale('log')
        # Set major ticks only (e.g. 10^2, 10^1, 10^0, 10^-1, etc.)
        plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        # Disable minor ticks
        plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
        plt.xlabel('Iteration')
        plt.ylabel('Total Variation\ndistance')
        plt.tight_layout()

        # create the save directory
        save_dir = Path(self.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / 'tv_loss.png')
        plt.close()


    def save_all_dists(self,
        y: torch.Tensor, 
        As: torch.Tensor,
        pred_params: List, 
        num_cells: Optional[int] = 1000, 
        grid_limit: Optional[float] = 50.0
    ):

        # initialize grid for approximate KL and FI calculations
        x1 = torch.linspace(-grid_limit + 0.5, grid_limit - 0.5, num_cells).to(torch.float32)
        x2 = torch.linspace(-grid_limit + 0.5, grid_limit - 0.5, num_cells).to(torch.float32)
        x1, x2 = torch.meshgrid(x1, x2, indexing='ij')
        grid = torch.stack([x1.ravel(), x2.ravel()], dim=-1)

        # create the destination folder
        dest_dir = Path(self.save_dir) / 'dist_plots'
        dest_dir.mkdir(parents=True, exist_ok=True)

        num_exps = len(pred_params)
        for exp_no in range(num_exps):
            
            # calculate the ground truth posterior log-probs
            mus, covs, pis = self.mus.cpu(), self.covs.cpu(), self.pis.cpu()
            log_prior = self.log_gmm(grid, mus, covs, pis)
            log_likelihood = self.log_likelihood(y, grid, As[exp_no])
            log_gt_posterior = log_prior + log_likelihood 

            # calculate the estimated (LMC) posterior log-probs
            lmc_params = pred_params[exp_no]
            log_lmc_posterior = self.log_gmm(grid, lmc_params['mus'], lmc_params['covs'], lmc_params['pis'])
            
            # Plot
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            titles = [r'$-\l og \ell(y|x)$', r'$-\log p(x)$', r'$-\log \pi(x|y)$', r'$-\log \nu(x|y)$']
            images = [-log_likelihood, -log_prior, -log_gt_posterior, -log_lmc_posterior]
            images = images[:-2] + [img + torch.logsumexp(-img, dim=0) for img in images[-2:]]
            cmaps = ['Blues_r', 'Oranges_r', 'PuRd_r', 'PuRd_r']
            vmaxs = [10, 10, 20, 20]
            vmins = [0.1, 0.1, 2.0, 2.0]
            #vmaxs = [None,] * 4
            #vmins = [None,] * 4

            for ax, data, title, cmap, vmin, vmax in zip(axs, images, titles, cmaps, vmins, vmaxs):
                data = data.reshape(num_cells, num_cells).cpu().numpy()
                im = ax.imshow(data, origin='lower', extent=[-50, 50, -50, 50],
                            cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=12)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(dest_dir / f'exp{exp_no}.png')  # Saves in the current working directory


def main():
    parser = argparse.ArgumentParser(description="Langeinv Monte Carlo sampling script.")

    # experiment parameters
    parser.add_argument('--num_exps', type=int, default=1, help="Number of experiments to run")
    parser.add_argument('--num_iter', type=int, default=2000, help="Number of iterations to run LMC")
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of  samples to simulate LMC')
    parser.add_argument('--num_cells', type=int, default=1000, help="Number of cells to discretize the grid")
    parser.add_argument('--device', type=str, default="cpu", choices=("cuda", "cpu"), help="device to run the experiments")
    parser.add_argument('--grid_limit', type=int, default=50, help="Grid limit for calculating the loss")
    parser.add_argument('--save_dir', type=str, default='./results', help='directory path for saving results')
    
    # annealing parameters
    parser.add_argument('--sigmin', type=float, default=0.0, help="Minimum sigma value for ALMC perturbation level")
    parser.add_argument('--sigma0', type=float, default=10.0, help="maximum/initial sigma value")
    parser.add_argument('--alpha0', type=float, default=10.0, help="maximum annealing parameter")
    parser.add_argument('--epsmax', type=float, default=2.5, help="Maximum synthetic score prior error")

    # inverse problem parameters 
    parser.add_argument('--y_std', type=float, default=1.0, help="standard deviation of measurement noise")
    parser.add_argument('--noise_type', type=str, default='gauss', help='measurement noise modeling in the inverse problem')

    # LMC algorithm parameters
    parser.add_argument('--std_noise_grad', type=float, default=0.0, help="Standard deviation of the SFO")
    parser.add_argument('--gamma', type=float, default=0.05, help="Step size for LMC algorithm")
    parser.add_argument('--estimator', type=str, default="sgd", choices=("sgd", "sgde", "sgdm", "page", "storm", "eve", "zo_sgd"), help="estimator to choose form score ll calculation")

    # hyperparameters for optimization algorithms
    parser.add_argument('--eve_beta1', type=float, default=0.999, help="beta1 parameter for EVE")
    parser.add_argument('--eve_beta2', type=float, default=-1.001, help="beta2 parameter for EVE")
    parser.add_argument('--beta', type=float, default=0.99, help="momentum parameter for SGD with momentum (SGDm)")
    parser.add_argument('--batch_size', type=int, default=10, help="large batch size for PAGE-type algorithms")
    parser.add_argument('--mini_batch_size', type=int, default=1, help="mini-batch size for all algorithms")

    # zeroth-order smoothing parameter
    parser.add_argument('--mu', type=float, default=1e-3, help='zeroth-order smoothing parameter')
    args = parser.parse_args()

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    match args.estimator:
        case 'sgdm':
            # beta: 0.99, mini_batch_size: 6 (default)
            estimator = SGDm(beta=args.beta, mini_batch_size=args.batch_size)
        case 'sgde':
            # beta (1-p): 0.4, batch_size: 10
            estimator = SGDe(beta=args.beta, batch_size=args.batch_size)
        case 'storm':
            # beta: 0.99, mini_batch_size: 3 (default)
            estimator = STORM(beta=args.beta, mini_batch_size=args.mini_batch_size)
        case 'page':
            # beta (1-p): 0.98, batch_size: 100, mini_batch_size: 2 (default)
            estimator = PAGE(beta=args.beta, batch_size=args.batch_size, mini_batch_size=args.mini_batch_size)
        case 'eve':
            estimator = EVE(beta1=args.eve_beta1, beta2=args.eve_beta2)
        case 'zo_sgd':
            estimator = ZO_SGD(p=1-args.beta, mu=args.mu, batch_size=args.batch_size, mini_batch_size=args.mini_batch_size)
        case _:
            # mini_batch_size: 6
            estimator = SGD(mini_batch_size=args.batch_size)
            print(f'default estimator SGD is chosen!')

    lmc = LangevinMonteCarlo(
        num_exps=args.num_exps, num_iter=args.num_iter, 
        num_samples=args.num_samples, num_cells=args.num_cells,
        grid_limit=args.grid_limit,
        sigma0=args.sigma0, alpha0=args.alpha0,
        sigmin=args.sigmin, epsmax=args.epsmax,
        std_noise_grad=args.std_noise_grad,
        gamma=args.gamma, 
        y_std=args.y_std, estimator=estimator,
        device=args.device, save_dir=args.save_dir,
        noise_type=args.noise_type
    )

    # run the experiments
    print("LMC is being run...")
    exps, forward_ops, y = lmc.run_apmc_red()
    particles, grads = exps['particles'], exps['grads']
    print("Experiments are completed!")

    # check for 'nan' of 'inf' values!
    has_nan = torch.isnan(particles).any()
    if has_nan:
        print("particles include nan values!")
    has_inf = torch.isinf(particles).any()
    if has_inf:
        print("particles include inf values!")

    # calculate the FI, KL, and TV losses for each experiment
    print("Losses are being calculated...")
    results = lmc.calculate_losses(particles, forward_ops, y)
    print("Losses are calculated!")

    # save the plot of all distributions
    lmc.save_all_dists(y, forward_ops, results['pred_gmm'])
    print("all distributions are saved and plotted!\n" + "-" * 20)

    # set the path for different results
    kl_save_path = Path(args.save_dir) / 'kl_loss.npy'
    fi_save_path = Path(args.save_dir) / 'fi_loss.npy'
    tv_save_path = Path(args.save_dir) / 'tv_loss.npy'
    grads_save_path = Path(args.save_dir) / 'grads_used.npy'

    # stack all results
    kl_loss = torch.stack(results['KL'], dim=0)
    fi_loss = torch.stack(results['FI'], dim=0)
    tv_loss = torch.stack(results['TV'], dim=0)

    # save all the results
    np.save(kl_save_path, kl_loss.numpy())
    np.save(fi_save_path, fi_loss.numpy())
    np.save(tv_save_path, tv_loss.numpy())
    np.save(grads_save_path, grads.numpy())
    print(f"KL, FI, and TV losses are saved to {args.save_dir}")

    # save the loss plots
    lmc.save_losses(kl_loss, fi_loss, tv_loss)
    print(f"Loss plots are saved to {args.save_dir}")

# run the algorithm 
if __name__ == '__main__':
    main()
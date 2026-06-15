
import sys
import math
import copy 
import json
import math
import random
import numpy as np
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Tuple, Optional, List, Dict

# PyTorch imports
import torch
from torch.func import grad
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt 
from utils import compute_posterior, save_stat_imgs, save_test_img_y
from estimators import SGDm, SGDe, STORM, PAGE, SGD

class LangevinMonteCarlo:
    
    """
    This class is designed for the numerical validation of Langevin Monte Carlo for 
    posterior sampling. 
    """
    
    def __init__(
        self, 
        estimator: object,
        num_iter: int = 1000,
        num_samples: int = 1000,
        sigmin: float = 0.0, 
        sigma0: float = 10.0, 
        alpha0: float = 10.0, 
        y_std: float = 1.0, 
        std_noise_grad: float = 0.0,
        gamma: float = 0.8,
        seed: int = 42,
        prior_params: Optional[Dict[str, torch.Tensor]] = None,
        test_img: Optional[torch.Tensor] = None,
        xi: float = 0.975, # THIS WAS 0.975
        device : str = 'cpu',
        save_dir: str | Path = './results_val',
        save_freq: Optional[int] = 500,
        y_dim: Optional[int] = 307, 
        noise_type: Optional[str] = 'gauss'
    ):
        assert device in ('cuda', 'cpu')

        # estimator parameters 
        self.estimator_dict = {'estimator_name': estimator.__class__.__name__}
        self.estimator_dict.update(estimator.__dict__)

        # experiment parameters
        self.num_iter = num_iter 
        self.num_samples = num_samples
        self.device = device
        self.rng = torch.Generator(device).manual_seed(seed)
        self.seed = seed
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.std_noise_grad = std_noise_grad

        # annealing hyperparameters
        self.xi = xi 
        self.sigma0 = sigma0
        self.alpha0 = alpha0
        self.sigmin = sigmin
        self.gamma = gamma 

        # inverse problem parameters
        self.y_std = y_std
        self.y_dim = y_dim
        self.noise_type = noise_type 
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_config(save_dir / 'configs.json')

        # likelihood score function estimator
        self.estimator = estimator
        
        # get sigma and alpha schedules
        self.sigmas, self.alphas = self.get_sigma_alpha_schedule()

        # prior parameters
        self.prior_params = prior_params
        self.prior_params['precision'] = torch.linalg.inv(self.prior_params['cov'])
        self.prior_params = {key: value.to(device) for key, value in self.prior_params.items()}

        # sample test image if not entered 
        if test_img is None:
            self.test_img = torch.distributions.MultivariateNormal(
                loc=self.prior_params['mean'], covariance_matrix=self.prior_params['cov']
            ).sample((1,)).squeeze().to(device)
        else:
            self.test_img = test_img.to(device)

        # select noise
        y_noise = torch.normal(
            mean=0.0, 
            std=self.y_std, 
            size=(y_dim,),
            generator=self.rng,
            device=self.device,
        )

        # pick a random forward model 
        x_dim = self.prior_params['mean'].shape[-1]
        self.A = torch.randn(y_dim, x_dim, device=self.device, generator=self.rng) * 1e-1  #THIS MUST BE 1E-1 (OR 1E-2)
        y = self.test_img @ self.A.T + y_noise
        
        self.y = y.squeeze()
        self.posterior_params = compute_posterior(self.A, self.y, self.prior_params, self.y_std)
        print("Analytical posterior was calculated!")

        # save prior images 
        save_stat_imgs(
            self.prior_params['mean'].cpu(),
            self.prior_params['cov'].cpu(),
            self.save_dir / 'prior_stats',
            mean_cmap='inferno',
            var_cmap='viridis'
        )

        # save posterior images
        save_stat_imgs(
            self.posterior_params['mean'].cpu(),
            self.posterior_params['cov'].cpu(),
            self.save_dir / 'post_stats',
            mean_cmap='inferno',
            var_cmap='viridis'
        )

    def save_config(self, path: str):
        def convert(o):
            # Convert non-serializable types
            if isinstance(o, (torch.device, Path)):
                return str(o)
            if isinstance(o, torch.Generator):
                return "torch.Generator"  # or None or str(o)
            return o

        config = {k: convert(v) for k, v in self.__dict__.items()}
        
        # Only include simple data, skip things like tensors or other objects
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
        
    def get_sigma_alpha_schedule(self):
        sigmas = torch.max(self.sigma0 *  self.xi ** torch.arange(self.num_iter), torch.tensor(self.sigmin)) # this was min before but changed to max as it makes more sense
        alphas = torch.max(self.alpha0 * sigmas ** 2, torch.tensor(1.0))
        return sigmas.to(self.device), alphas.to(self.device)

    def log_likelihood(self, 
        y: torch.Tensor, 
        x: torch.Tensor, 
        A: torch.Tensor
    )->torch.Tensor:

        """calculates the log-likelihood of the forward model with Gaussian noise"""

        assert y.device == x.device == A.device, "all variables must be on the same device!"

        # y : (1, m)
        # A : (m, d)
        # x: (N, d)

        # calculate the log-likelihood for each gaussian likelihood
        if y.ndim == 1:
            y = y.unsqueeze(0)              # (1, m)

        means = x @ A.T                     # (N, m)
        var = torch.ones_like(means) * self.y_std ** 2

        log_2pi = torch.log(torch.tensor(2 * torch.pi)).unsqueeze(0).unsqueeze(1).to(y.device)
        log_prob = -0.5 * ((y - means) ** 2 / var + torch.log(var) + log_2pi) # 

        return log_prob.sum(dim=1)

    def log_prior(self, x, mu, cov, inv_cov=None):
        """
        x:      (..., d)
        mu:     (d,)
        cov:    (d, d)
        inv_cov (optional): (d, d), if you want to reuse precomputed inverse
        """
        d = mu.shape[0]

        if inv_cov is None:
            inv_cov = torch.inverse(cov)

        # Center
        diff = x - mu  # (..., d)

        # Quadratic form (x-mu)^T Σ^{-1} (x-mu)
        # result shape: (...)
        m2 = torch.einsum('...i,ij,...j->...', diff, inv_cov, diff)

        # log(det Σ)
        # (use slogdet for numerical stability)
        sign, logdet = torch.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite with positive determinant.")

        # Normalization constant
        log_norm = -0.5 * (d * math.log(2 * math.pi) + logdet)

        # Full log-density
        return log_norm - 0.5 * m2

    def score_prior(self, x, mu, cov, inv_cov=None):
        """
        Computes ∇_x log N(x | mu, cov).

        x:        (..., d)
        mu:       (d,)
        cov:      (d, d)
        inv_cov:  (d, d), optional precomputed inverse
        returns:  (..., d)
        """
        if inv_cov is None:
            inv_cov = torch.inverse(cov)

        diff = x - mu  # (..., d)
        # score = - Σ^{-1} (x - μ)
        score = -torch.einsum('ij,...j->...i', inv_cov, diff)
        return score


    def score_likelihood(self, y: torch.Tensor, x: torch.Tensor, A: torch.Tensor):
        '''calculates the score function of the likelihood (y = Ax + e) for data points, x'''

        # SHAPES
        # y : (1, m)
        # A : (m, d)
        # x: (N, d)

        residual = y[None] - (x @ A.T)
        return residual @ A / self.y_std ** 2

    def run_apmc_red(self):

        x_dim = self.prior_params['mean'].shape[-1]
        # initialize the samples
        x_init = torch.rand(
            size=(self.num_samples, x_dim),
            device=self.device, 
            generator=self.rng,
        ) * 6 - 3 # sample from U[-3, 3]

        # generate multiple seeds
        seeds = torch.randint(0, 2**63 - 1, (4,), device=self.rng.device, generator=self.rng)
        # Create new generators from these seeds
        [gen_eps, gen_sig, gen_ll, gen_z] = [torch.Generator(self.device).manual_seed(int(seed)) for seed in seeds]

        particles = []
        x_k, x_k_1, x_k_2 = x_init.clone().detach(), None, None
        score_ll_est, score_ll_est_1, score_ll_est_2 = None, None, None
        prior_mean, prior_cov = self.prior_params['mean'], self.prior_params['cov']
        g_k = torch.tensor(0, dtype=torch.float32, device=self.device)
        for k in tqdm(range(self.num_iter), desc=f"γ={self.gamma}, σ_min={self.sigmin:.3f}"):
    
            # generate the perturbed/smoothed covariance matrix
            perturbed_cov = prior_cov + self.sigmas[k] ** 2 * torch.eye(prior_cov.shape[-1], device=self.device)
            inv_cov = torch.linalg.inv(perturbed_cov).to(self.device)

            # calculate perturbed score function of the prior (in practice, this is score-based generative model)
            score_k = self.score_prior(x_k, prior_mean, perturbed_cov, inv_cov)
            score_has_nan = torch.isnan(score_k).any()

            # calculate the update term
            score_ll_est = self.estimator.likelihood_update(
                x_k=x_k, x_k_1=x_k_1, x_k_2=x_k_2,
                y=self.y, A=self.A,
                m_k_1=score_ll_est_1, m_k_2=score_ll_est_2,
                std_noise_grad=self.std_noise_grad,
                y_std=self.y_std, noise_type=self.noise_type,
                rng=gen_ll
            )
            # update the score estimates
            score_ll_est_2 = score_ll_est_1.clone().detach() if score_ll_est_1 is not None else None
            score_ll_est_1 = score_ll_est.clone().detach()

            # update the iterate/particles
            x_k_2 = x_k_1.clone().detach() if x_k_1 is not None else None
            x_k_1 = x_k.clone().detach()

            # estimate of the score of the posterior
            score_post_est = score_ll_est - self.alphas[k] * score_k

            # do the APMC update
            z_k = torch.randn(size=x_k.shape, generator=gen_z, device=self.device)
            x_k = x_k - self.gamma * score_post_est + math.sqrt(2 * self.gamma) * z_k

            if torch.isnan(x_k).any():
                print("Nan value encountered in LMC sampling!")
                break

            if torch.isinf(x_k).any():
                print("Inf encountered in LMC sampling!")
                break
            
            if k % self.save_freq == 0:
                particles.append(x_k.detach().cpu().clone())

        return torch.stack(particles, dim=0)
    
    def save_estimated_stats(self, particle_list: torch.Tensor):
        n_estimates = len(particle_list)
        for idx in range(n_estimates):
            particles = particle_list[idx]
            mean = torch.mean(particles, dim=0)
            cov = torch.cov(particles.T)
            save_stat_imgs(mean, cov, self.save_dir / f'estimates/step_{idx+1}', mean_cmap='inferno', var_cmap='viridis', verbose=False)
def main():
    parser = argparse.ArgumentParser(description="Statistical Validation arguments")

    # experiment parameters
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--num_iter', type=int, default=5000, help='number of iterations')
    parser.add_argument('--num_exps', type=int, default=1, help='number of experiments to run')
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--mode_name', type=str, default='female', choices=('male', 'female'), help='type of mode, male or female')

    # LMC parameters
    parser.add_argument('--sigma0', type=float, default=192, help='initial sigma (smoothing level of prior) value')
    parser.add_argument('--alpha0', type=float, default=2.5, help='initial value of the alpha value (weight of the prior)')
    parser.add_argument('--gamma', type=float, default=5e-4, help='step size of LMC') # previously this was 1e-4 but 5e-4 is optimum I think
    parser.add_argument('--xi', type=float, default=0.975, help='flattening level for annealing')
    parser.add_argument('--num_samples', type=int, default=1000, help='number o of samples to run LMC')
    parser.add_argument('--sigmin', type=float, default=math.sqrt(1/4000), help='minimum sigma (smoothing) value')

    # inverse problem parameters
    parser.add_argument('--y_std', type=float, default=0.5, help='noise level in forward model') # DEFAULT WAS 0.1 BUT CHOOSE 0.5 FOR ZO EXPS
    parser.add_argument('--noise_type', type=str, default='gauss', help='type of the noise modeling in inverse problem')
    parser.add_argument('--test_idx', type=int, default=4, choices=(1,2,3,4,5), help='index of the test images to be used for posterior sampling')    

    # estimator parameters 
    parser.add_argument('--estimator', type=str, default="sgd", choices=("sgd", "sgde", "sgdm", "page", "storm"))

    parser.add_argument('--beta', type=float, default=0.99, help="momentum parameter for SGD with momentum (SGDm)")
    parser.add_argument('--batch_size', type=int, default=10, help="large batch size for PAGE-type algorithms")
    parser.add_argument('--mini_batch_size', type=int, default=1, help="mini-batch size for all algorithms")
    parser.add_argument('--std_noise_grad', type=float, default=0.0, help="standard deviation of stochastic first-order oracles") # DEFAULT WAS 7.5

    # reporting & saving results 
    parser.add_argument('--save_dir', type=str, default='./results', help='save directory of the experiment')
    parser.add_argument('--save_freq', type=int, default=500, help='frequency of reporting results at every <input> iteration')
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
        case _:
            # mini_batch_size: 6
            estimator = SGD(mini_batch_size=args.batch_size)
            print(f'default estimator SGD is chosen!')

    print("estimator name:", args.estimator)
    ROOT_PATH = Path("/scratch/gilbreth/sahinm/LangevinMonteCarlo/StochasticInverse/toy_experiments/statistical_validation")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # load the statistics of the GMM prior calculated over all CelebA dataset
    prior_stats_path = ROOT_PATH / 'datasets/gender_stats.pt'
    prior_params = torch.load(prior_stats_path)
    # rename the parameter names
    prior_params['mean'] = prior_params.pop(f'mean_{args.mode_name}')
    prior_params['cov'] = prior_params.pop(f'cov_{args.mode_name}')

    # read the test image for LMC sampling
    test_path = ROOT_PATH / f'test_imgs/test_img{args.test_idx}.npy'
    test_img = torch.from_numpy(np.load(test_path))

    # initialize Langevin Monte Carlo simulation object
    lmc = LangevinMonteCarlo(
        estimator=estimator,
        num_iter=args.num_iter,
        sigmin=args.sigmin, 
        sigma0=args.sigma0,
        alpha0=args.alpha0,
        gamma=args.gamma,
        std_noise_grad=args.std_noise_grad,
        y_std=args.y_std, 
        seed=args.seed, 
        prior_params=prior_params,
        test_img=test_img.flatten(),
        xi=args.xi,
        device=args.device,
        save_dir=args.save_dir,
        save_freq=args.save_freq, 
        y_dim=115, # 107 or 102 (this was 115) DEFAULT: 115
        noise_type=args.noise_type
    )

    # save the test image
    test_img_save_dir = lmc.save_dir / 'test_img_and_y'
    test_img_save_dir.mkdir(exist_ok=True, parents=True)
    save_test_img_y(test_img.cpu(), lmc.y.cpu(), lmc.A.cpu(), test_img_save_dir)
    # run the experiment for LMC posterior sampling
    particle_list = lmc.run_apmc_red()
    np.save(save_dir / 'particle_list.npy', particle_list.cpu().numpy())
    print(f"LMC statistical validation exp is completed and results are saved to {save_dir}!")

    lmc.save_estimated_stats(particle_list)
    print("All particles are saved!")


# run the algorithm 
if __name__ == '__main__':
    main()
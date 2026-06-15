import json
import os, pathlib, torch
import numpy as np
from typing import Optional, Iterable
from .base import BaseCallbackModule
from pmc.utils.compute_snr import compute_snr
from pmc.utils import losses
from pmc.utils.get_results import create_results_df
from collections import defaultdict
from torchmetrics.image import StructuralSimilarityIndexMeasure
#from skimage.metrics import structural_similarity

def nest_list_of_dicts(records):
    def recurse(dicts):
        out = defaultdict(list)
        for d in dicts:
            for k, v in d.items():
                out[k].append(v)
        # recurse where values are dicts
        for k, v in out.items():
            if isinstance(v[0], dict):
                out[k] = recurse(v)
        return dict(out)

    return recurse(records)

class LocalScalarCallbackModule(BaseCallbackModule):
    
    def __init__(
        self,
        vis_freq=1,
    ) -> None:
        super().__init__()
        self.vis_freq = vis_freq
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(-1, 1))

    def on_batch_start(self, module, batch, batch_idx):
        # initialize output dictionary at every batch start
        module.logger.init_dict()

    def on_batch_end(self, module, samples, means, stds, batch, batch_idx, add_info: dict=None):

        # log additional info
        if add_info is not None:
            module.logger.log_tensor(add_info)

        # set directory
        save_dir = os.path.join(module.cfg.exp_dir, f'batch{batch_idx}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        # save final mean and std
        x, y = batch
        xrecons, xdrift_recons = samples
        adjoint_recon = module.pmc.forward_model.adjoint(y)
        if torch.is_complex(adjoint_recon):
            adjoint_recon = adjoint_recon.real
        xrecon_mean, xdrift_recon_mean = means
        xrecon_std, xdrift_recon_std = stds
        module.logger.log_tensor({
            f'batch{batch_idx}_xrecons': xrecons,
            f'batch{batch_idx}_xdrift_recons': xdrift_recons,
            f'batch{batch_idx}_xrecon_means': xrecon_mean,
            f'batch{batch_idx}_xdrift_recon_means': xdrift_recon_mean,
            f'batch{batch_idx}_xrecon_stds': xrecon_std, 
            f'batch{batch_idx}_xdrift_recon_stds': xdrift_recon_std,
            f'batch{batch_idx}_adjoint_recons': adjoint_recon,
            f'batch{batch_idx}_x': x, 
            f'batch{batch_idx}_y': y
        })

        # load metrics to cuda
        self.ssim = self.ssim.to(x.device)

        # save final PSNR and SNR
        # Example: compute_snr(xpred, xtrue)
        adjoint_ssim = self.ssim(adjoint_recon.clip(-1,1), x)
        adjoint_psnr, adjoint_snr, adjoint_mse = compute_snr(adjoint_recon.clip(-1,1), x)

        xrecon_ssim = self.ssim(xrecon_mean.clip(-1,1), x)
        xrecon_psnr, xrecon_snr, xrecon_mse = compute_snr(xrecon_mean.clip(-1,1), x)
        xdrift_recon_ssim = self.ssim(xdrift_recon_mean.clip(-1,1),x)
        xdrift_recon_psnr, xdrift_recon_snr, xdrift_recon_mse = compute_snr(xdrift_recon_mean.clip(-1,1), x)
        module.logger.log_tensor({
            f'batch{batch_idx}_adjoint_ssim': adjoint_ssim,
            f'batch{batch_idx}_adjoint_psnr': adjoint_psnr,
            f'batch{batch_idx}_adjoint_snr': adjoint_snr,
            f'batch{batch_idx}_adjoint_mse': adjoint_mse,
            f'batch{batch_idx}_xrecon_ssim': xrecon_ssim,
            f'batch{batch_idx}_xrecon_psnr': xrecon_psnr, 
            f'batch{batch_idx}_xrecon_snr': xrecon_snr,
            f'batch{batch_idx}_xrecon_mse': xrecon_mse,
            f'batch{batch_idx}_xdrift_recon_ssim': xdrift_recon_ssim,
            f'batch{batch_idx}_xdrift_recon_psnr': xdrift_recon_psnr, 
            f'batch{batch_idx}_xdrift_recon_snr': xdrift_recon_snr,
            f'batch{batch_idx}_xdrift_recon_mse': xdrift_recon_mse,
        })

        # log memory consumption values
        mem_logger = module.pmc.mem_logger
        if mem_logger.enabled is True:
            mem_dict = nest_list_of_dicts(mem_logger.records)
            with open(os.path.join(save_dir, 'mem_logs.json'), 'w') as f:
                json.dump(mem_dict, f, indent=4)

        # save output dictionary every batch
        module.logger.to_npz(os.path.join(save_dir, 'outs'))

    def on_iteration_end(self, module, iteration_outputs, batch, batch_idx, t):
        x, y = batch
        x_t, x_t_drift, x_t_prev, drift, score, diffusion, df_grad, _ = iteration_outputs
        
        # set directory
        save_dir = os.path.join(module.cfg.exp_dir, f'batch{batch_idx}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        n_samples = x_t.shape[0]
        logger = module.logger

        if t % self.vis_freq == 0:
            
            # log xiter mean psnr & snr
            xiter_mean = x_t.mean(dim=0, keepdim=True)
            xiter_drift_mean = x_t_drift.mean(dim=0, keepdim=True)
            xiter_ssim = self.ssim(xiter_mean.clip(-1, 1), x)
            xiter_psnr, xiter_snr, xiter_mse = compute_snr(xiter_mean.clip(-1,1), x)
            xiter_drift_ssim = self.ssim(xiter_drift_mean.clip(-1, 1), x)
            xiter_drift_psnr, xiter_drift_snr, xiter_drift_mse = compute_snr(xiter_drift_mean.clip(-1,1), x)
            scalar_dict = {
                f"batch{batch_idx}_xiter_drift_SNR": xiter_drift_snr,
                f"batch{batch_idx}_xiter_drift_PSNR": xiter_drift_psnr,
                f"batch{batch_idx}_xiter_drift_MSE": xiter_drift_mse,
                f"batch{batch_idx}_xiter_drift_SSIM": xiter_drift_ssim,
                f"batch{batch_idx}_xiter_SNR": xiter_snr,
                f"batch{batch_idx}_xiter_PSNR": xiter_psnr,
                f"batch{batch_idx}_xiter_MSE": xiter_mse,
                f"batch{batch_idx}_xiter_SSIM": xiter_ssim,
            }
            logger.log_iter_tensor(scalar_dict, t)
            
            for sample_idx in range(n_samples):
                tensor_dict = {
                    f"batch{batch_idx}_dgrad_norm": df_grad[sample_idx].flatten().norm(),
                    f"batch{batch_idx}_score_norm": score[sample_idx].flatten().norm(),
                    #f"batch{batch_idx}_datafit": module.pmc.forward_model.eval(x_t_drift[sample_idx:sample_idx+1], y),
                    f"batch{batch_idx}_drift_norm": drift[sample_idx].flatten().norm(),
                    f"batch{batch_idx}_residual_norm": (x_t-x_t_prev)[sample_idx].flatten().norm(),
                }
                logger.log_iter_persample_tensor(tensor_dict, sample_idx, t)

                # SNR, PSNR
                x_ssim = self.ssim(x_t[sample_idx,None].clip(-1,1), x).squeeze()
                x_psnr, x_snr, x_mse = compute_snr(x_t[sample_idx].clip(-1,1), x[0])
                xdrift_ssim = self.ssim(x_t_drift[sample_idx,None].clip(-1,1), x).squeeze()
                xdrift_psnr, xdrift_snr, xdrift_mse = compute_snr(x_t_drift[sample_idx].clip(-1,1), x[0])
                scalar_dict = {
                    f"batch{batch_idx}_xdrift_SNR": xdrift_snr,
                    f"batch{batch_idx}_xdrift_PSNR": xdrift_psnr,
                    f"batch{batch_idx}_xdrift_MSE": xdrift_mse,
                    f"batch{batch_idx}_xdrift_SSIM": xdrift_ssim,
                    f"batch{batch_idx}_x_SNR": x_snr,
                    f"batch{batch_idx}_x_PSNR": x_psnr,
                    f"batch{batch_idx}_x_MSE": x_mse,
                    f"batch{batch_idx}_x_SSIM": x_ssim
                }
                logger.log_iter_persample_tensor(scalar_dict, sample_idx, t)
    
    def on_inference_end(self, module):
        # create a spreadsheet for results and save it to the results dir
        _ = create_results_df(module.cfg.exp_dir)

'''
class LocalScalarCallbackModule(BaseCallbackModule):
    def __init__(
        self,
        vis_freq=1,
    ) -> None:
        super().__init__()
        self.vis_freq = vis_freq

    def on_batch_start(self, module, batch, batch_idx):
        # initialize output dictionary at every batch start
        module.logger.init_dict()

    def on_batch_end(self, module, samples, means, stds, batch, batch_idx):
        # set directory
        save_dir = os.path.join(module.cfg.exp_dir, f'batch{batch_idx}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        # save final mean and std
        x, y = batch
        xrecons, xdrift_recons = samples
        adjoint_recon = module.pmc.forward_model.adjoint(y)
        if torch.is_complex(adjoint_recon):
            adjoint_recon = adjoint_recon.real
        xrecon_mean, xdrift_recon_mean = means
        xrecon_std, xdrift_recon_std = stds
        module.logger.log_tensor({
            f'batch{batch_idx}_xrecons': xrecons,
            f'batch{batch_idx}_xdrift_recons': xdrift_recons,
            f'batch{batch_idx}_xrecon_means': xrecon_mean,
            f'batch{batch_idx}_xdrift_recon_means': xdrift_recon_mean,
            f'batch{batch_idx}_xrecon_stds': xrecon_std, 
            f'batch{batch_idx}_xdrift_recon_stds': xdrift_recon_std,
            f'batch{batch_idx}_adjoint_recons': adjoint_recon,
            f'batch{batch_idx}_x': x, 
            f'batch{batch_idx}_y': y
        })
        # save final PSNR and SNR
        # Example: compute_snr(xpred, xtrue)
        adjoint_psnr, adjoint_snr = compute_snr(adjoint_recon.clip(-1,1), x)
        xrecon_psnr, xrecon_snr = compute_snr(xrecon_mean.clip(-1,1), x)
        xdrift_recon_psnr, xdrift_recon_snr = compute_snr(xdrift_recon_mean.clip(-1,1), x)
        module.logger.log_tensor({
            f'batch{batch_idx}_adjoint_psnr': adjoint_psnr,
            f'batch{batch_idx}_adjoint_snr': adjoint_snr,
            f'batch{batch_idx}_xrecon_psnr': xrecon_psnr, 
            f'batch{batch_idx}_xrecon_snr': xrecon_snr,
            f'batch{batch_idx}_xdrift_recon_psnr': xdrift_recon_psnr, 
            f'batch{batch_idx}_xdrift_recon_snr': xdrift_recon_snr,
        })

        # log memory consumption values
        mem_logger = module.pmc.mem_logger
        if mem_logger.enabled is True:
            mem_dict = nest_list_of_dicts(mem_logger.records)
            with open(os.path.join(save_dir, 'mem_logs.json'), 'w') as f:
                json.dump(mem_dict, f, indent=4)

        # save output dictionary every batch
        module.logger.to_npz(os.path.join(save_dir, 'outs'))

    def on_iteration_end(self, module, iteration_outputs, batch, batch_idx, t):
        x, y = batch
        x_t, x_t_drift, x_t_prev, drift, score, diffusion, df_grad = iteration_outputs
        
        # set directory
        save_dir = os.path.join(module.cfg.exp_dir, f'batch{batch_idx}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        n_samples = x_t.shape[0]
        logger = module.logger

        if t % self.vis_freq == 0:
            
            # log xiter mean psnr & snr
            xiter_mean = x_t.mean(dim=0, keepdim=True)
            xiter_drift_mean = x_t_drift.mean(dim=0, keepdim=True)
            xiter_psnr, xiter_snr = compute_snr(xiter_mean.clip(-1,1), x)
            xiter_drift_psnr, xiter_drift_snr = compute_snr(xiter_drift_mean.clip(-1,1), x)
            scalar_dict = {
                f"batch{batch_idx}_xiter_drift_SNR": xiter_drift_snr,
                f"batch{batch_idx}_xiter_drift_PSNR": xiter_drift_psnr,
                f"batch{batch_idx}_xiter_SNR": xiter_snr,
                f"batch{batch_idx}_xiter_PSNR": xiter_psnr,
            }
            logger.log_iter_tensor(scalar_dict, t)
            
            for sample_idx in range(n_samples):
                tensor_dict = {
                    f"batch{batch_idx}_dgrad_norm": df_grad[sample_idx].flatten().norm(),
                    f"batch{batch_idx}_score_norm": score[sample_idx].flatten().norm(),
                    f"batch{batch_idx}_datafit": module.pmc.forward_model.eval(x_t_drift[sample_idx:sample_idx+1], y),
                    f"batch{batch_idx}_drift_norm": drift[sample_idx].flatten().norm(),
                    f"batch{batch_idx}_residual_norm": (x_t-x_t_prev)[sample_idx].flatten().norm(),
                }
                logger.log_iter_persample_tensor(tensor_dict, sample_idx, t)

                # SNR, PSNR
                x_psnr, x_snr = compute_snr(x_t[sample_idx].clip(-1,1), x[0])
                xdrift_psnr, xdrift_snr = compute_snr(x_t_drift[sample_idx].clip(-1,1), x[0])
                scalar_dict = {
                    f"batch{batch_idx}_xdrift_SNR": xdrift_snr,
                    f"batch{batch_idx}_xdrift_PSNR": xdrift_psnr,
                    f"batch{batch_idx}_x_SNR": x_snr,
                    f"batch{batch_idx}_x_PSNR": x_psnr,
                }
                logger.log_iter_persample_tensor(scalar_dict, sample_idx, t)
'''
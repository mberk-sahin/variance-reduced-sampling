import os 
import sys
import re
import json
import pandas as pd
import torch
from pathlib import Path
import numpy as np
import statistics
#import losses
from pmc.utils import losses
from typing import Optional, List

METRICS = ['PSNR', 'SSIM', 'SNR', 'NRMSE', 'SD', 'MSE', 'NLL', 'MEM']

def _natural_batch_sort_key(p: Path) -> int:
    # batch12 -> 12 (falls back to 0 if no digits)
    m = re.search(r"\d+", p.name)
    return int(m.group()) if m else 0

def _load_metrics_from_dict(batch_dir: Path):

    # get memory metric
    mem_data_path = str(batch_dir / 'mem_logs.json')
    with open(mem_data_path, "r") as f:
        mem_data = json.load(f)
    mem_used = statistics.mean(mem_data['df_grad_est']['peak_mib']) + statistics.mean(mem_data['score']['peak_mib'])
    
    # get reconstruction metrics
    result_path = batch_dir / 'outs.npz'
    result_data = np.load(result_path, allow_pickle=True)['arr_0'].item()
    batch_name = batch_dir.name
    
    # get PSNR, SNR, SSIM, NRMSE
    psnr, snr = result_data[f"{batch_name}_xrecon_psnr"].item(), result_data[f"{batch_name}_xrecon_snr"]
    mse, ssim = result_data[f"{batch_name}_xrecon_mse"].item(), result_data[f"{batch_name}_xrecon_ssim"]
    # get ground truth and preds 
    xrecons, xreal = torch.from_numpy(result_data[f"{batch_name}_xrecons"]), torch.from_numpy(result_data[f"{batch_name}_x"])
    nrmse = losses.nrmse(xrecons, xreal).mean()
    # calculate negative log-likelihood (NLL)
    xrecons_mean, xrecons_std = xrecons.mean(dim=0, keepdim=True), xrecons.std(dim=0, keepdim=True)
    nll_i = (xrecons - xrecons_mean) ** 2 / (2 * xrecons_std ** 2) + 0.5 * torch.log(2 * torch.pi * xrecons_std ** 2)
    nll = nll_i.mean()
    sd = xrecons_std.mean()

    # results dict
    results_dict = {
        'batch_idx': int(batch_name.rstrip().split('batch')[-1]),
        'PSNR': psnr,
        'SSIM': ssim,
        'SNR': snr,
        'NRMSE': nrmse,
        'SD': sd,
        'MSE': mse,
        'NLL': nll,
        'MEM': mem_used
    }
    results_dict = {k: v.item() if torch.is_tensor(v) or isinstance(v, np.ndarray) else v for k, v in results_dict.items()}
    return results_dict

def create_results_df(results_dir : str | Path, out_path: Optional[str | Path]=None, max_batch: int = None):
    results_dir = Path(results_dir)
    results_list = sorted(results_dir.glob("batch*"), key=_natural_batch_sort_key)[:max_batch]
    nresults = len(results_list)

    # initialize the dataframe to generate spreadsheet
    columns = ['batch_idx',] + METRICS
    df = pd.DataFrame(np.nan, index=range(nresults), columns=columns, dtype=float)

    # fill each row from its batch folder 
    for i, batch_dir in enumerate(results_list):
        results_dict = _load_metrics_from_dict(batch_dir)
        # save the results to the spreadsheet
        df.loc[i, results_dict.keys()] = list(results_dict.values())
    
    # sort rows by batch idx 
    df = df.sort_values("batch_idx").reset_index(drop=True)
    if out_path is None:
        out_path = results_dir / 'out.xlsx'
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    df.to_excel(out_path, index=False)
    print(
        "-"*20 
        + f"\nResults are saved to {str(out_path)}!"
        + "\n" + "-"*20 
    )
    return df

def cumulative_results(
    results_dir_list: List[str],
    out_path: str | Path, 
    method_name_list: Optional[List[str]]=None,
    max_batch: Optional[int]=-1,
    use_existing_df: Optional[bool]=None
):
    n_methods = len(results_dir_list)
    columns = ['method_name',] + METRICS
    df = pd.DataFrame(np.nan, index=range(n_methods), columns=columns, dtype=float)

    # get the cumulative results
    for i, results_dir in enumerate(results_dir_list):
        # get results for the method
        if use_existing_df is False:
            results = create_results_df(results_dir, max_batch=max_batch)
        else:
            results_path = os.path.join(results_dir, 'out.xlsx')
            if os.path.exists(results_path):
                results = pd.read_excel(results_path)
            else:
                print(f"There is no spreadsheet with name 'out.xlsx'under {results_path}! Skipping...")
                continue
            
        result_series = results.mean(axis=0).copy().drop('batch_idx')
        result_series['method_name'] = method_name_list[i] if method_name_list is not None else f'method{i}'
        df.loc[i, result_series.index] = result_series.values
    
    df.to_excel(out_path, index=False)
    return df
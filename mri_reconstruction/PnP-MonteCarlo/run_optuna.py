import json
import os
import re
import subprocess
import argparse 
from pathlib import Path
import pandas as pd
import functools

import optuna

PYTHON = "python"
RUN_SCRIPT = "run_pmc.py"
CONFIG_SGD   = "/scratch/gautschi/sahinm/LangevinMonteCarlo/mri_reconstruction/PnP-MonteCarlo/configs/radial_mri/sgd.yaml"
RESULTS_DIR = "/scratch/gautschi/sahinm/LangevinMonteCarlo/mri_reconstruction/PnP-MonteCarlo/results"

def run_once(config_path: str, exp_name: str, overrides: dict) -> float:
    """
    Runs run_pmc.py with overrides and returns PSNR.
    Assumes the run produces a JSON metrics file at: <exp_dir>/metrics.json
    OR prints 'METRIC psnr=...' to stdout.
    """
    cmd = [PYTHON, "-u", RUN_SCRIPT, "--config", config_path, "--exp_name", exp_name]
    for k, v in overrides.items():
        cmd += [k, str(v)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed:\n{proc.stderr}\n{proc.stdout}")

    metrics_path = Path(RESULTS_DIR) / exp_name / "out.xlsx"
    if metrics_path.exists():
        df = pd.read_excel(metrics_path)
        return df['PSNR'].mean()

    raise RuntimeError("Could not find PSNR spreadsheet.")

def objective(trial: optuna.Trial, save_dir: Path, args: argparse.ArgumentParser) -> float:
    # ---- you choose what you search over ----
    # Example: cost budget C and betas. Batch sizes derived from C.
    #C = trial.suggest_categorical("cost_budget", [2, 4, 8, 16, 32])

    # get the extra arguments to override
    batch_size = args.batch_size
    num_lines = args.num_lines

    # find a beta parameter for each algorithm
    beta_sgdm = trial.suggest_float("beta_sgdm", 0.1, 0.99)
    beta_storm = trial.suggest_float("beta_storm", 0.1, 0.99)

    # find a step size for each algorithm 
    step_sgdm = trial.suggest_float("step_sgdm", 5e-7, 5e-5, log=True)
    step_storm = trial.suggest_float("step_storm", 5e-7, 5e-5, log=True)

    # Ensure unique exp_name per trial
    if trial.number == 0:
        psnr_sgd = run_once(
            CONFIG_SGD,
            exp_name=f"{save_dir.name}/sgd",
            overrides={
                "--model.estimator.name": "SGD",        # or "SGD"
                "--model.estimator.batch_size": batch_size,
                "--model.gamma": 5e-06,                 # fixed as you said
                "--model.estimator.noise_type": "coil",
                "--model.forward_model.num_lines": args.num_lines
            },
        )
    else:
        exp_name = f"{save_dir}/sgd/out.xlsx"
        df = pd.read_excel(exp_name)
        psnr_sgd = df['PSNR'].mean().item()

    exp_name_sgdm = (
        f"{save_dir.name}/trial{trial.number}"
        f"_sgdm_beta{beta_sgdm:.2f}"
        f"_step{step_sgdm:.1e}"
    )
    psnr_sgdm = run_once(
        CONFIG_SGD,
        exp_name=exp_name_sgdm,
        overrides={
            "--model.estimator.name": "SGDm",
            "--model.estimator.batch_size": batch_size,
            "--model.estimator.beta": beta_sgdm,
            "--model.gamma": step_sgdm,    # if shared/fixed in your framework
            "--model.estimator.noise_type": "coil",
            "--model.forward_model.num_lines": args.num_lines
        },
    )

    exp_name_storm = (
        f"{save_dir.name}/trial{trial.number}"
        f"_storm_beta{beta_storm:.2f}"
        f"_step{step_storm:.1e}"
    )
    psnr_storm = run_once(
        CONFIG_SGD,
        exp_name=exp_name_storm,
        overrides={
            "--model.estimator.name": "STORM",
            "--model.estimator.batch_size": batch_size,
            "--model.estimator.beta": beta_storm,
            "--model.gamma": step_storm,
            "--model.estimator.noise_type": "coil",
            "--model.forward_model.num_lines": args.num_lines
        },
    )

    # ---- enforce ordering + maximize separation ----
    # Want SGD < SGDp < PAGE and biggest gap.
    # Penalty if violated:
    penalty = max(0.0, psnr_sgd - psnr_sgdm) + max(0.0, psnr_sgdm - psnr_storm)

    # Score: encourage big separation, but punish violations strongly
    score = (psnr_storm - psnr_sgd) - 100.0 * penalty

    # Log helpful stuff
    trial.set_user_attr("psnr_sgd", psnr_sgd)
    trial.set_user_attr("psnr_sgdm", psnr_sgdm)
    trial.set_user_attr("psnr_storm", psnr_storm)

    trial.set_user_attr("gamma_sgd", 5e-06)
    trial.set_user_attr("gamma_sgdm", step_sgdm)
    trial.set_user_attr("gamma_storm", step_storm)

    trial.set_user_attr("beta_sgdm", beta_sgdm)
    trial.set_user_attr("beta_storm", beta_storm)
    return score

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finding the best beta and gamma parameters for SGDm and STORM.')
    parser.add_argument(
        "--batch_size", "-b",
        type=int, default=32,
        help="number of gradient calculation per-step"
    )

    parser.add_argument(
        "--dirname",
        type=str,
        default="optuna_study",
        help="name of the directory to save the experiment results"
    )

    parser.add_argument(
        "--num_trials",
        type=int, 
        default=5,
        help="number of trials to run optuna"
    )

    parser.add_argument(
        "--num_lines",
        type=int,
        default=32,
        help="number of lines to sample (decides acceleration factor)"
    )
    
    args = parser.parse_args()

    study_dir = Path(RESULTS_DIR) / args.dirname
    study_dir.mkdir(parents=True, exist_ok=True)
    storage_path = "sqlite:///" + str((study_dir / 'results.db'))

    print(
        "-"*20 +
        f"\nbatch size: {args.batch_size}\n"
        f"dirname: {args.dirname}\n"
        f"num_trials: {args.num_trials}\n"
        + "-"*20
    )
    
    study = optuna.create_study(
        direction="maximize",
        study_name="sgd_sgdm_storm",
        storage=storage_path,
        load_if_exists=True
    )
    objective = functools.partial(objective, save_dir=study_dir, args=args)
    study.optimize(objective, n_trials=args.num_trials)

    print("Best trial:", study.best_trial.number)
    print("Best params:", study.best_trial.params)
    print("PSNRs:",
          study.best_trial.user_attrs["psnr_sgd"],
          study.best_trial.user_attrs["psnr_sgdm"],
          study.best_trial.user_attrs["psnr_storm"])

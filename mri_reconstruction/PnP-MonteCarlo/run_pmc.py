import os, argparse, torch
from pmc.config import Configurator

parser = argparse.ArgumentParser(description='Autonomous Diffusion Model (ADM)')
parser.add_argument(
    "--config", "-c", 
    type=str, 
    help="Path to config file"
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    # parse_known_args lets us keep unknown flags like --model.estimator.batch_size
    args, unknown = parser.parse_known_args()

    # Convert: ["--model.estimator.batch_size", "16", "--model.gamma", "1e-5"]
    # into:   ["model.estimator.batch_size", "16", "model.gamma", "1e-5"]
    if len(unknown) % 2 != 0:
        raise ValueError(
            f"Override args must come in KEY VALUE pairs. Got odd number of tokens: {unknown}"
        )

    overrides = []
    i = 0
    while i < len(unknown):
        key = unknown[i]
        val = unknown[i + 1]
        if not key.startswith("--"):
            raise ValueError(f"Expected override key to start with '--', got: {key}")
        overrides.append(key[2:])  # strip leading "--"
        overrides.append(val)
        i += 2

    # attach to args
    args.overrides = overrides
    return args

if __name__ == '__main__':
    # parse arguments
    #args = parser.parse_args()
    args = parse_args()
    
    # configurate and save configuration file
    cc = Configurator(args)
    os.makedirs(cc.cfg.exp_dir, exist_ok=True)
    with open(f'{cc.cfg.exp_dir}/config.yaml', 'w') as f:
        f.write(str(cc.cfg))

    # initialize all modules
    exp, model, dataloader, callbacks = cc.init_all()

    # run
    exp(model, dataloader, callbacks)
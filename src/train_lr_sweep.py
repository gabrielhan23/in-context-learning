import os
import yaml
import uuid
import numpy as np
import argparse
from schema import load_config, dict_to_namespace
from train import main as train_main
from train import loss_history
import torch

LR_LIST = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

def run_sweep(base_config_path):
    curves = {}
    
    for lr in LR_LIST:
        print(f"Training with LR = {lr}")
        config_dict = load_config(base_config_path)
        args = dict_to_namespace(config_dict)
        args.training.learning_rate = lr
        args.training.train_steps = 1000
        run_id = f"lr_{lr}_{uuid.uuid4()}"
        out_dir = os.path.join(args.out_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)
        args.out_dir = out_dir

        config_save_path = os.path.join(out_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(config_dict, f)
        loss_history.clear()

        train_main(args, config_dict)

        curves[lr] = np.array(loss_history)

        np.save(os.path.join(out_dir, "loss.npy"), curves[lr])
        print(f"[INFO] Completed LR = {lr}. Saved loss to {out_dir}/loss.npy")

    return curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    all_curves = run_sweep(args.config)
    np.save("lr_sweep_curves.npy", all_curves)
    print("[INFO] Saved all curves to lr_sweep_curves.npy")

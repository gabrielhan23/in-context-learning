# Schema definitions for configuration validation
# Defines allowed tasks and default values

import os
import yaml
import argparse
from copy import deepcopy


TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
]

# Default configuration values
DEFAULT_CONFIG = {
    "model": {
        "family": None,  # required: "gpt2" or "lstm"
        "n_positions": None,  # required: maximum context length
        "n_dims": None,  # required: latent dimension
        "n_embd": None,  # required
        "n_layer": None,  # required
        "n_head": None,  # required
    },
    "training": {
        "task": None,  # required: one of TASK_LIST
        "task_kwargs": {},
        "num_tasks": None,
        "num_training_examples": None,
        "data": "gaussian",
        "batch_size": 64,
        "learning_rate": 3e-4,
        "train_steps": 1000,
        "save_every_steps": 1000,
        "keep_every_steps": -1,
        "resume_id": None,
        "curriculum": {
            "dims": {
                "start": None,  # required
                "end": None,  # required
                "inc": None,  # required
                "interval": None,  # required
            },
            "points": {
                "start": None,  # required
                "end": None,  # required
                "inc": None,  # required
                "interval": None,  # required
            },
        },
    },
    "wandb": {
        "project": "in-context-training",
        "entity": "in-context",
        "notes": "",
        "name": None,
        "log_every_steps": 10,
    },
    "out_dir": None,  # required
    "test_run": False,
}


def deep_update(base_dict, update_dict):
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(config_path, _apply_defaults=True):
    """Load YAML config with support for inheritance.
    
    Args:
        config_path: Path to the YAML config file
        _apply_defaults: Internal flag to control when defaults are applied
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if 'inherit' in config:
        inherit_list = config.pop('inherit')
        if not isinstance(inherit_list, list):
            inherit_list = [inherit_list]
        
        # Start with empty config for inheritance chain
        merged_config = {}
        
        # Load and merge inherited configs (without defaults)
        config_dir = os.path.dirname(config_path)
        for inherit_path in inherit_list:
            if inherit_path:  # Skip None or empty strings
                full_inherit_path = os.path.join(config_dir, inherit_path)
                if os.path.exists(full_inherit_path):
                    inherited_config = load_config(full_inherit_path, _apply_defaults=False)
                    deep_update(merged_config, inherited_config)
        
        # Merge current config on top of inherited configs
        deep_update(merged_config, config)
        config = merged_config
    
    # Apply defaults only at the top level (when _apply_defaults=True)
    if _apply_defaults:
        final_config = deepcopy(DEFAULT_CONFIG)
        deep_update(final_config, config)
        config = final_config
    
    return config


def dict_to_namespace(d):
    """Convert nested dictionary to nested namespace for attribute access."""
    if isinstance(d, dict):
        namespace = argparse.Namespace()
        for key, value in d.items():
            setattr(namespace, key, dict_to_namespace(value))
        return namespace
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def validate_config(config):
    """Validate configuration values."""
    # Check model family
    if config.get("model", {}).get("family") not in ["gpt2", "lstm"]:
        raise ValueError("model.family must be 'gpt2' or 'lstm'")
    
    # Check task
    if config.get("training", {}).get("task") not in TASK_LIST:
        raise ValueError(f"training.task must be one of {TASK_LIST}")
    
    # Check data
    if config.get("training", {}).get("data") not in ["gaussian"]:
        raise ValueError("training.data must be 'gaussian'")
    
    return config

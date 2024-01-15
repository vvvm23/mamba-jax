import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import equinox as eqx
import jax
import numpy as np
import torch
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from loguru import logger

import wandb


def seed_others(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def wandb_init(args):
    return wandb.init(
        project="mamba-jax",
        config=vars(args),
        mode=None if args.wandb else "disabled",
    )


def setup_sharding(args):
    devices = mesh_utils.create_device_mesh((len(jax.devices()),))
    logger.info(devices)
    sharding = PositionalSharding(devices)
    return sharding, len(devices)


def make_experiment_directory(args):
    experiments_root = Path("experiments")

    exp_id = "mamba-jax_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_root = experiments_root / exp_id

    exp_root.mkdir(parents=True)

    return exp_root


def update_metrics(metrics: Dict[str, jax.Array], running_metrics: Optional[Dict[str, float]] = None):
    if running_metrics is None:
        running_metrics = {k: 0.0 for k in metrics}

    for k, v in metrics.items():
        if hasattr(v, "item"):
            v = v.item()

        running_metrics[k] += v

    return running_metrics


def consolidate_metrics(metrics: Dict[str, jax.Array], step: int, prefix: str):
    for k, v in metrics.items():
        metrics[k] = v / step

    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    return metrics, None


def save_checkpoint(args, exp_dir, name, model, opt_state=None):
    path = exp_dir / f"checkpoint_{name}.eqx"
    logger.info(f"Saving checkpoint to '{path}'..")
    start_time = time.time()
    eqx.tree_serialise_leaves(path, model)
    logger.info(f"Finished saving in {time.time() - start_time:.3f} seconds")

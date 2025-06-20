import os

import hydra

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.metrics import NewsRecommenderMetrics
from utils.saving import get_output_run_dir
from utils.logging import setup_logging, console
from utils.model import initialize_model_and_dataset
from utils.orchestration import training_loop_orchestrator
from utils.logging import setup_wandb_session
from utils.device import setup_device


def setup_tensorflow_precision(use_mixed_precision: bool):
    """Sets the global TensorFlow precision policy."""
    if use_mixed_precision:
        console.log("Setting TensorFlow mixed precision policy to 'mixed_float16'.")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        console.log("Using TensorFlow default precision policy 'float32'.")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main_training_entry(cfg: DictConfig) -> None:
    """Main entry point for training, configured by Hydra."""
    # Setup logging with rich formatting
    setup_logging(level=cfg.logging.level if hasattr(cfg.logging, "level") else "INFO")

    console.log(f"--- {cfg.model_name} Training Run Initializing ---")
    console.log("Configuration used:")
    console.log(OmegaConf.to_yaml(cfg))
    console.log("------------------------------------")

    # Setup device configuration
    setup_device(
        gpu_ids=cfg.device.gpu_ids if hasattr(cfg.device, "gpu_ids") else [],
        memory_limit=cfg.device.memory_limit if hasattr(cfg.device, "memory_limit") else 0.9
    )
    # Determine precision from the boolean flag
    use_mixed = cfg.device.mixed_precision if hasattr(cfg.device, "mixed_precision") else False
    
    # Set global TensorFlow precision policy
    setup_tensorflow_precision(use_mixed)

    # Set random seeds
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Initialize WandB
    setup_wandb_session(cfg)

    # Model and Dataset Initialization
    nrms_model, dataset_provider = initialize_model_and_dataset(cfg)

    # Metrics Calculator
    metrics_engine = NewsRecommenderMetrics(
        **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
    )

    # Setup output directory based on Hydra's current run path
    output_run_dir = get_output_run_dir(cfg)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"All outputs for this run will be saved in: {output_run_dir.resolve()}")

    # Rich Progress Bar context manager
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("({task.completed} of {task.total} batches)"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as global_progress_bar:

        training_loop_orchestrator(
            nrms_model,
            dataset_provider,
            cfg,
            metrics_engine,
            global_progress_bar,
            output_run_dir,
        )

    console.log(f"--- {cfg.model_name} Training Run Finished ---")


if __name__ == "__main__":
    main_training_entry()

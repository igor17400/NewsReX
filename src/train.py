import os

import hydra

# Set JAX as Keras backend before importing keras
os.environ["KERAS_BACKEND"] = "jax"

import keras

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

from src.utils.metrics.functions import NewsRecommenderMetrics
from src.utils.io.saving import get_output_run_dir
from src.utils.io.logging import setup_logging, console
from src.utils.model.model import initialize_model_and_dataset
from src.utils.training.orchestration import training_loop_orchestrator
from src.utils.metrics.wrapper import create_news_metrics, LightweightNewsMetrics
from src.utils.io.logging import setup_wandb_session
from src.utils.device.device import setup_device


def setup_precision(use_mixed_precision: bool):
    """Sets the global Keras precision policy."""
    if use_mixed_precision:
        console.log("Setting Keras mixed precision policy to 'mixed_float16'.")
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
    else:
        console.log("Using Keras default precision policy 'float32'.")


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

    # Set global Keras precision policy
    setup_precision(use_mixed)

    # Set random seeds for all backends (Keras 3 handles backend-specific seeding)
    keras.utils.set_random_seed(cfg.seed)

    # Initialize WandB
    setup_wandb_session(cfg)

    # Prepare training metrics
    if LightweightNewsMetrics.should_use_lightweight_metrics(cfg):
        # Use lightweight metrics during training, custom metrics in callbacks
        training_metrics = LightweightNewsMetrics.create_training_metrics()
        console.log("Using lightweight metrics during training with custom metrics in callbacks")
    else:
        # Use full custom metrics during training (slower but more comprehensive)
        training_metrics = create_news_metrics(
            NewsRecommenderMetrics(**cfg.metrics.params if hasattr(cfg.metrics, "params") else {}))
        console.log("Using full custom metrics during training")

    # Model and Dataset Initialization (includes compilation with metrics)
    model, dataset_provider = initialize_model_and_dataset(cfg, training_metrics)

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
            model,
            dataset_provider,
            cfg,
            metrics_engine,
            global_progress_bar,
            output_run_dir,
        )

    console.log(f"--- {cfg.model_name} Training Run Finished ---")


if __name__ == "__main__":
    main_training_entry()

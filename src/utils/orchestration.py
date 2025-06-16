from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import wandb

import tensorflow as tf
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress

from .metrics import NewsRecommenderMetrics
from .engine import train_step_fn
from .evaluation import (
    _run_initial_validation,
    _run_epoch_evaluation,
    _run_final_testing,
    get_main_comparison_metric,
)
from .logging import (
    log_metrics_to_console_fn,
    log_epoch_summary_fn,
    log_metrics_to_wandb_fn,
)

from .saving import save_run_summary_fn

console = Console()


def calculate_num_batches(dataset_size: int, batch_size: int) -> int:
    """Calculate the number of batches based on dataset size and batch size."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    return (dataset_size + batch_size - 1) // batch_size


def train_epoch_fn(
    model: tf.keras.Model,
    dataloader_train: tf.data.Dataset,
    num_batches_epoch: int,
    progress_bar: Progress,
    log_interval_steps: int,
    epoch_idx: int,
) -> Dict[str, float]:
    """Trains the model for one epoch."""

    for m in model.metrics:
        m.reset_state()

    sum_batch_losses = 0.0
    epoch_train_task = progress_bar.add_task(
        f"Epoch {epoch_idx + 1}", total=num_batches_epoch, loss="N/A"
    )

    for batch_count, data_batch in enumerate(dataloader_train, 1):
        batch_step_metrics = train_step_fn(model, data_batch)
        current_batch_loss = float(batch_step_metrics["loss"])
        sum_batch_losses += current_batch_loss

        # Only update progress bar and log batch loss
        progress_bar.update(
            epoch_train_task,
            advance=1,
            description=f"Epoch {epoch_idx + 1} (Batch Loss: {current_batch_loss:.4f})",
        )

        if batch_count % log_interval_steps == 0:
            console.log(
                f"Epoch {epoch_idx+1}, Batch {batch_count}/{num_batches_epoch}, "
                f"Batch Loss: {current_batch_loss:.4f}."
            )

    progress_bar.remove_task(epoch_train_task)

    avg_loss = sum_batch_losses / num_batches_epoch if num_batches_epoch > 0 else 0.0

    return {"loss": avg_loss}


def update_best_epoch_metrics(
    best_epoch_summary: Dict[str, Any],
    current_epoch_idx: int,
    current_train_metrics_epoch: Dict[str, float],
    current_val_metrics_epoch: Dict[str, float],
    min_improvement: float = 0.01,
) -> bool:
    """Updates the summary of the best epoch if current epoch is better.

    Args:
        best_epoch_summary: Dictionary to store the best epoch information
        current_epoch_idx: Current epoch index
        current_train_metrics_epoch: Training metrics for current epoch
        current_val_metrics_epoch: Validation metrics for current epoch
        min_improvement: Minimum improvement required to consider it better

    Returns:
        bool: True if this is a new best epoch, False otherwise
    """
    # Get current average metric and check if improvement is significant
    current_avg_metric, is_significant = get_main_comparison_metric(
        validation_metrics=current_val_metrics_epoch,
        min_improvement=min_improvement,
    )

    previous_best = best_epoch_summary.get("average_metric_value", -float("inf"))

    # Check if this is a new best epoch
    is_new_best = current_avg_metric > previous_best and is_significant

    if is_new_best:
        best_epoch_summary.clear()
        best_epoch_summary |= {
            "epoch_number": current_epoch_idx + 1,
            "train_loss_at_best": float(current_train_metrics_epoch.get("loss", float("nan"))),
            "val_loss_at_best": float(current_val_metrics_epoch.get("loss", float("nan"))),
            "average_metric_value": current_avg_metric,
            **{f"val_{k}": v for k, v in current_val_metrics_epoch.items()},
        }
        console.log(
            f"[bold green]New best epoch! Average metric improved from {previous_best:.4f} to {current_avg_metric:.4f}[/bold green]"
        )
    else:
        console.log(
            f"[yellow]No improvement. Current average: {current_avg_metric:.4f}, Previous best: {previous_best:.4f}[/yellow]"
        )

    return is_new_best


def _setup_training_directories(output_directory: Path, model_name: str) -> Tuple[Path, Path, Path]:
    """
    Setup directories for models and predictions.

    Args:
        output_directory: Path to the output directory
        model_name: Name of the model
    Returns:
        Tuple[Path, Path, Path]: Path to the best model, path to the last model, and path to the predictions
    """
    models_save_dir = output_directory / "models"
    models_save_dir.mkdir(parents=True, exist_ok=True)
    predictions_save_dir = output_directory / "predictions"
    predictions_save_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_save_dir / f"{model_name}_best.weights.h5"
    last_model_path = models_save_dir / f"{model_name}_last.weights.h5"

    return best_model_path, last_model_path, predictions_save_dir


def _handle_epoch_end(
    model: tf.keras.Model,
    epoch_idx: int,
    final_train_metrics: Dict[str, float],
    final_val_metrics: Dict[str, float],
    best_epoch_metrics: Dict[str, Any],
    cfg: DictConfig,
    best_model_path: Path,
    last_model_path: Path,
    wandb_history: Optional[Dict[str, List[float]]],
) -> Tuple[bool, int]:
    """Handle end of epoch operations: metrics update, model saving, early stopping."""
    is_new_best = update_best_epoch_metrics(
        best_epoch_metrics,
        epoch_idx,
        final_train_metrics,
        final_val_metrics,
        cfg.train.early_stopping.min_improvement,
    )

    log_epoch_summary_fn(
        epoch_idx,
        final_train_metrics,
        final_val_metrics,
        is_new_best,
        wandb_history,
    )

    if is_new_best:
        model.save_weights(str(best_model_path))
        patience = cfg.train.early_stopping.patience
    else:
        patience = best_epoch_metrics.get("patience", cfg.train.early_stopping.patience) - 1
        best_epoch_metrics["patience"] = patience

    model.save_weights(str(last_model_path))
    return is_new_best, patience


def training_loop_orchestrator(
    model: tf.keras.Model,
    dataset_provider: Any,
    cfg: DictConfig,
    custom_metrics_engine: NewsRecommenderMetrics,
    progress_bar_manager: Progress,
    output_directory: Path,
) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
    """Orchestrates the main training loop, including validation, early stopping, and model saving."""
    # Initial setup
    num_train_batches = calculate_num_batches(dataset_provider.train_size, cfg.train.batch_size)
    best_epoch_metrics = {
        "average_metric_value": -float("inf"),
        "patience": cfg.train.early_stopping.patience,
    }
    wandb_history = {} if cfg.logging.enable_wandb else None

    # Setup directories
    best_model_path, last_model_path, predictions_save_dir = _setup_training_directories(
        output_directory, model.name
    )

    # Initial validation
    initial_metrics = _run_initial_validation(
        model, dataset_provider, custom_metrics_engine, progress_bar_manager, cfg
    )
    if wandb.run and wandb_history is not None:
        log_metrics_to_wandb_fn(
            {f"initial_val/{k}": v for k, v in initial_metrics.items()},
            0,
            wandb_history,
        )

    # Main training loop
    console.log("[bold]Starting training loop...[/bold]")
    dataloader_train = dataset_provider.train_dataloader(batch_size=cfg.train.batch_size)
    overall_task = progress_bar_manager.add_task(
        "Overall Training Progress", total=cfg.train.num_epochs
    )

    for epoch_idx in range(cfg.train.num_epochs):
        progress_bar_manager.update(
            overall_task,
            description=f"Training Epoch {epoch_idx+1}/{cfg.train.num_epochs}",
        )

        # Training and validation
        train_metrics = train_epoch_fn(
            model,
            dataloader_train,
            num_train_batches,
            progress_bar_manager,
            cfg.logging.log_every_n_steps,
            epoch_idx,
        )
        val_metrics = _run_epoch_evaluation(
            model,
            dataset_provider,
            custom_metrics_engine,
            progress_bar_manager,
            cfg,
            epoch_idx,
            predictions_save_dir,
        )

        # Handle epoch end
        _, patience = _handle_epoch_end(
            model,
            epoch_idx,
            train_metrics,
            val_metrics,
            best_epoch_metrics,
            cfg,
            best_model_path,
            last_model_path,
            wandb_history,
        )

        if patience <= 0:
            console.log(
                f"[bold red]Early stopping triggered after epoch {epoch_idx + 1}.[/bold red]"
            )
            break

        progress_bar_manager.update(overall_task, advance=1)

    progress_bar_manager.remove_task(overall_task)

    # Final testing
    test_metrics = (
        _run_final_testing(
            model,
            dataset_provider,
            custom_metrics_engine,
            progress_bar_manager,
            cfg,
            best_model_path,
            last_model_path,
            best_epoch_metrics,
            predictions_save_dir,
        )
        if cfg.eval.run_test_after_training
        else None
    )

    # Save results and cleanup
    if test_metrics:
        log_metrics_to_console_fn(test_metrics, "Final Test")
        if wandb.run and wandb_history is not None:
            log_metrics_to_wandb_fn(
                {f"test/{k}": v for k, v in test_metrics.items()},
                cfg.train.num_epochs + 1,
                wandb_history,
            )

    save_run_summary_fn(
        output_directory, cfg, initial_metrics, best_epoch_metrics, test_metrics, wandb_history
    )

    if wandb.run:
        wandb.summary |= {
            "best_epoch": best_epoch_metrics.get("epoch_number"),
            "best_val_metric": best_epoch_metrics.get("average_metric_value"),
            **{f"best/{k}": v for k, v in best_epoch_metrics.items()},
        }
        wandb.finish()
        console.log("Wandb session finished.")

    return best_epoch_metrics, test_metrics

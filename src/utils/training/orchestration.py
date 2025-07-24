from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import wandb

import keras
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress

from src.utils.metrics.functions import NewsRecommenderMetrics
from .callbacks import FastEvaluationCallback, SlowEvaluationCallback
from ..evaluation.evaluation import (
    _run_initial_validation,
    _run_final_testing,
)
from ..io.logging import (
    log_metrics_to_console_fn,
    log_metrics_to_wandb_fn,
)
from ..io.saving import save_run_summary_fn

console = Console()


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


class RichProgressCallback(keras.callbacks.Callback):
    """Keras callback to integrate with Rich progress bars."""

    def __init__(self, progress_manager: Progress, num_epochs: int):
        super().__init__()
        self.progress_manager = progress_manager
        self.num_epochs = num_epochs
        self.overall_task = None
        self.epoch_task = None

    def on_train_begin(self, logs=None):
        """Initialize overall progress tracking."""
        self.overall_task = self.progress_manager.add_task(
            "Overall Training Progress",
            total=self.num_epochs
        )

    def on_epoch_begin(self, epoch, logs=None):
        """Initialize epoch progress tracking."""
        if hasattr(self.model, 'train_dataset') and hasattr(self.model.train_dataset, '__len__'):
            try:
                total_steps = len(self.model.train_dataset)
            except:
                total_steps = None
        else:
            total_steps = None

        self.epoch_task = self.progress_manager.add_task(
            f"Epoch {epoch + 1}/{self.num_epochs}",
            total=total_steps,
            visible=True
        )

    def on_batch_end(self, batch, logs=None):
        """Update epoch progress after each batch."""
        if self.epoch_task is not None:
            loss = logs.get('loss', 0.0) if logs else 0.0
            self.progress_manager.update(
                self.epoch_task,
                advance=1,
                description=f"Epoch {self.model._epoch + 1}/{self.num_epochs} (Loss: {loss:.4f})"
            )

    def on_epoch_end(self, epoch, logs=None):
        """Clean up epoch progress and update overall progress."""
        if self.epoch_task is not None:
            self.progress_manager.remove_task(self.epoch_task)
            self.epoch_task = None

        if self.overall_task is not None:
            self.progress_manager.update(
                self.overall_task,
                advance=1,
                description=f"Completed Epoch {epoch + 1}/{self.num_epochs}"
            )

    def on_train_end(self, logs=None):
        """Clean up overall progress tracking."""
        if self.overall_task is not None:
            self.progress_manager.remove_task(self.overall_task)


def training_loop_orchestrator(
        model: keras.Model,
        dataset_provider: Any,
        cfg: DictConfig,
        custom_metrics_engine: NewsRecommenderMetrics,
        progress_bar_manager: Progress,
        output_directory: Path,
) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
    """Orchestrates training using Keras 3 model.fit() with custom callbacks."""

    # Setup directories
    best_model_path, last_model_path, predictions_save_dir = _setup_training_directories(
        output_directory, model.name
    )

    # Setup WandB history tracking
    wandb_history = {} if cfg.logging.enable_wandb else None

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

    # Prepare training and validation datasets
    console.log("[bold]Preparing datasets for Keras 3 training...[/bold]")
    train_dataset = dataset_provider.train_dataloader(batch_size=cfg.train.batch_size)

    # Setup callbacks
    callbacks = []

    # Add Rich progress callback
    progress_callback = RichProgressCallback(progress_bar_manager, cfg.train.num_epochs)
    callbacks.append(progress_callback)

    # Add evaluation callback (fast or slow)
    if cfg.eval.fast_evaluation:
        eval_callback = FastEvaluationCallback(
            dataset_provider=dataset_provider,
            custom_metrics_engine=custom_metrics_engine,
            cfg=cfg,
            progress_bar_manager=progress_bar_manager,
            predictions_save_dir=predictions_save_dir,
            wandb_history=wandb_history,
        )
    else:
        eval_callback = SlowEvaluationCallback(
            dataset_provider=dataset_provider,
            custom_metrics_engine=custom_metrics_engine,
            cfg=cfg,
            progress_bar_manager=progress_bar_manager,
            predictions_save_dir=predictions_save_dir,
            wandb_history=wandb_history,
        )

    eval_callback.set_model_save_path(best_model_path)
    callbacks.append(eval_callback)

    # Add ModelCheckpoint callback for last model
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(last_model_path),
        save_weights_only=True,
        save_freq='epoch',
        verbose=0
    )
    callbacks.append(checkpoint_callback)

    # Add ReduceLROnPlateau if configured
    if hasattr(cfg.train, 'reduce_lr_on_plateau') and cfg.train.reduce_lr_on_plateau.enabled:
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=cfg.train.reduce_lr_on_plateau.factor,
            patience=cfg.train.reduce_lr_on_plateau.patience,
            min_lr=cfg.train.reduce_lr_on_plateau.min_lr,
            verbose=1
        )
        callbacks.append(reduce_lr_callback)

    # Verify model is compiled (should always be true after model initialization)
    if not model.compiled:
        raise RuntimeError(
            "Model is not compiled! This should not happen as the model should be "
            "compiled during initialization in initialize_model_and_dataset(). "
            "Please check the model initialization process."
        )

    # Main training using model.fit()
    console.log("[bold]Starting Keras 3 model.fit() training...[/bold]")

    try:
        history = model.fit(
            train_dataset,
            epochs=cfg.train.num_epochs,
            callbacks=callbacks,
            verbose=0,  # We handle progress with Rich
            validation_data=None,  # We handle validation with custom callbacks
        )

        console.log("[bold green]Training completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.log("[bold red]Training interrupted by user.[/bold red]")
        history = None
    except Exception as e:
        console.log(f"[bold red]Training failed with error: {e}[/bold red]")
        raise

    # Get best epoch metrics from the evaluation callback
    best_epoch_metrics = eval_callback.best_epoch_metrics

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
        wandb.summary.update({
            "best_epoch": best_epoch_metrics.get("epoch_number"),
            "best_val_metric": best_epoch_metrics.get("average_metric_value"),
            **{f"best/{k}": v for k, v in best_epoch_metrics.items()},
        })
        wandb.finish()
        console.log("Wandb session finished.")

    return best_epoch_metrics, test_metrics

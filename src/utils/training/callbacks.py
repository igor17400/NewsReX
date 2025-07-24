from pathlib import Path
from typing import Dict, Any, Optional
import wandb

import keras
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress

from src.utils.metrics.functions import NewsRecommenderMetrics
from ..io.logging import log_metrics_to_console_fn, log_metrics_to_wandb_fn

console = Console()


class FastEvaluationCallback(keras.callbacks.Callback):
    """Custom Keras callback for fast evaluation during training."""

    def __init__(
            self,
            dataset_provider: Any,
            custom_metrics_engine: NewsRecommenderMetrics,
            cfg: DictConfig,
            progress_bar_manager: Progress,
            predictions_save_dir: Optional[Path] = None,
            wandb_history: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dataset_provider = dataset_provider
        self.custom_metrics_engine = custom_metrics_engine
        self.cfg = cfg
        self.progress_bar_manager = progress_bar_manager
        self.predictions_save_dir = predictions_save_dir
        self.wandb_history = wandb_history or {}

        # Best epoch tracking
        self.best_epoch_metrics = {
            "average_metric_value": -float("inf"),
            "patience": cfg.train.early_stopping.patience,
        }
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Run fast evaluation at the end of each epoch."""
        if not self.cfg.eval.fast_evaluation:
            return

        # Create mode-specific directory for predictions
        mode_specific_dir = self.predictions_save_dir / "val" if self.predictions_save_dir else None
        if mode_specific_dir:
            mode_specific_dir.mkdir(parents=True, exist_ok=True)

        # Run fast evaluation
        val_metrics = self.model.fast_evaluate(
            user_hist_dataloader=self.dataset_provider.user_history_dataloader(mode="val"),
            impression_iterator=self.dataset_provider.impression_dataloader(mode="val"),
            news_dataloader=self.dataset_provider.news_dataloader(),
            metrics_calculator=self.custom_metrics_engine,
            progress=self.progress_bar_manager,
            mode="val",
            save_predictions_path=mode_specific_dir if self.cfg.eval.save_predictions else None,
            epoch=epoch,
        )

        # Log metrics to console
        log_metrics_to_console_fn(val_metrics, f"Epoch {epoch + 1} Validation")

        # Log to WandB if enabled
        if wandb.run and self.wandb_history is not None:
            log_metrics_to_wandb_fn(
                {f"val/{k}": v for k, v in val_metrics.items()},
                epoch + 1,
                self.wandb_history,
            )

        # Update best epoch metrics and check for early stopping
        is_new_best = self._update_best_epoch_metrics(epoch, val_metrics)

        if is_new_best:
            self.wait = 0
            # Save best model weights
            if hasattr(self, 'best_model_path'):
                self.model.save_weights(str(self.best_model_path))
        else:
            self.wait += 1

        # Check for early stopping
        if self.wait >= self.cfg.train.early_stopping.patience:
            console.log(f"[bold red]Early stopping triggered after epoch {epoch + 1}.[/bold red]")
            self.model.stop_training = True

    def _update_best_epoch_metrics(
            self,
            epoch_idx: int,
            val_metrics: Dict[str, float],
            min_improvement: float = 0.01
    ) -> bool:
        """Update best epoch metrics tracking."""
        # Calculate main comparison metric (average of AUC, MRR, nDCG@5, nDCG@10)
        main_metrics = ["auc", "mrr", "ndcg@5", "ndcg@10"]
        metric_values = []

        for metric in main_metrics:
            value = val_metrics.get(metric)
            if value is not None:
                metric_values.append(float(value))

        if not metric_values:
            console.log(
                "[yellow]No main metrics found in validation metrics. Using 0.0 as fallback.[/yellow]"
            )
            return False

        current_avg_metric = sum(metric_values) / len(metric_values)
        previous_best = self.best_epoch_metrics.get("average_metric_value", -float("inf"))

        # Check if this is a new best epoch
        is_new_best = current_avg_metric > previous_best and current_avg_metric > min_improvement

        if is_new_best:
            self.best_epoch_metrics.clear()
            self.best_epoch_metrics.update({
                "epoch_number": epoch_idx + 1,
                "train_loss_at_best": float(self.model.history.history.get("loss", [0.0])[-1]),
                "val_loss_at_best": float(val_metrics.get("loss", float("nan"))),
                "average_metric_value": current_avg_metric,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })
            console.log(
                f"[bold green]New best epoch! Average metric improved from {previous_best:.4f} to {current_avg_metric:.4f}[/bold green]"
            )
        else:
            console.log(
                f"[yellow]No improvement. Current average: {current_avg_metric:.4f}, Previous best: {previous_best:.4f}[/yellow]"
            )

        return is_new_best

    def set_model_save_path(self, best_model_path: Path):
        """Set the path where the best model should be saved."""
        self.best_model_path = best_model_path


class SlowEvaluationCallback(keras.callbacks.Callback):
    """Custom Keras callback for traditional batch-wise evaluation during training."""

    def __init__(
            self,
            dataset_provider: Any,
            custom_metrics_engine: NewsRecommenderMetrics,
            cfg: DictConfig,
            progress_bar_manager: Progress,
            predictions_save_dir: Optional[Path] = None,
            wandb_history: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dataset_provider = dataset_provider
        self.custom_metrics_engine = custom_metrics_engine
        self.cfg = cfg
        self.progress_bar_manager = progress_bar_manager
        self.predictions_save_dir = predictions_save_dir
        self.wandb_history = wandb_history or {}

        # Best epoch tracking
        self.best_epoch_metrics = {
            "average_metric_value": -float("inf"),
            "patience": cfg.train.early_stopping.patience,
        }
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Run slow evaluation at the end of each epoch."""
        if self.cfg.eval.fast_evaluation:
            return

        from .evaluation import run_evaluation_epoch

        # Create mode-specific directory for predictions
        mode_specific_dir = self.predictions_save_dir / "val" if self.predictions_save_dir else None
        if mode_specific_dir:
            mode_specific_dir.mkdir(parents=True, exist_ok=True)

        # Run traditional evaluation
        val_metrics = run_evaluation_epoch(
            self.model,
            self.dataset_provider.val_dataloader(batch_size=self.cfg.eval.batch_size),
            self.custom_metrics_engine,
            self.dataset_provider.val_size,
            self.progress_bar_manager,
            mode="val",
            save_predictions_dir=mode_specific_dir if self.cfg.eval.save_predictions else None,
            epoch_idx=epoch,
        )

        # Log metrics to console
        log_metrics_to_console_fn(val_metrics, f"Epoch {epoch + 1} Validation")

        # Log to WandB if enabled
        if wandb.run and self.wandb_history is not None:
            log_metrics_to_wandb_fn(
                {f"val/{k}": v for k, v in val_metrics.items()},
                epoch + 1,
                self.wandb_history,
            )

        # Update best epoch metrics and check for early stopping
        is_new_best = self._update_best_epoch_metrics(epoch, val_metrics)

        if is_new_best:
            self.wait = 0
            # Save best model weights
            if hasattr(self, 'best_model_path'):
                self.model.save_weights(str(self.best_model_path))
        else:
            self.wait += 1

        # Check for early stopping
        if self.wait >= self.cfg.train.early_stopping.patience:
            console.log(f"[bold red]Early stopping triggered after epoch {epoch + 1}.[/bold red]")
            self.model.stop_training = True

    def _update_best_epoch_metrics(
            self,
            epoch_idx: int,
            val_metrics: Dict[str, float],
            min_improvement: float = 0.01
    ) -> bool:
        """Update best epoch metrics tracking."""
        # Calculate main comparison metric (average of AUC, MRR, nDCG@5, nDCG@10)
        main_metrics = ["auc", "mrr", "ndcg@5", "ndcg@10"]
        metric_values = []

        for metric in main_metrics:
            value = val_metrics.get(metric)
            if value is not None:
                metric_values.append(float(value))

        if not metric_values:
            console.log(
                "[yellow]No main metrics found in validation metrics. Using 0.0 as fallback.[/yellow]"
            )
            return False

        current_avg_metric = sum(metric_values) / len(metric_values)
        previous_best = self.best_epoch_metrics.get("average_metric_value", -float("inf"))

        # Check if this is a new best epoch
        is_new_best = current_avg_metric > previous_best and current_avg_metric > min_improvement

        if is_new_best:
            self.best_epoch_metrics.clear()
            self.best_epoch_metrics.update({
                "epoch_number": epoch_idx + 1,
                "train_loss_at_best": float(self.model.history.history.get("loss", [0.0])[-1]),
                "val_loss_at_best": float(val_metrics.get("loss", float("nan"))),
                "average_metric_value": current_avg_metric,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })
            console.log(
                f"[bold green]New best epoch! Average metric improved from {previous_best:.4f} to {current_avg_metric:.4f}[/bold green]"
            )
        else:
            console.log(
                f"[yellow]No improvement. Current average: {current_avg_metric:.4f}, Previous best: {previous_best:.4f}[/yellow]"
            )

        return is_new_best

    def set_model_save_path(self, best_model_path: Path):
        """Set the path where the best model should be saved."""
        self.best_model_path = best_model_path


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

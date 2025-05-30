import os

# Set TensorFlow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import datetime

from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import hydra
import tensorflow as tf
import wandb
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.device import setup_device
from utils.logging import setup_logging
from utils.metrics import NewsRecommenderMetrics
from utils.losses import get_loss

# Setup logging
setup_logging()

console = Console()
rich_handler = RichHandler(console=console, rich_tracebacks=True, show_time=True)


def setup_wandb(cfg: DictConfig) -> None:
    """Initialize Weights & Biases logging for experiment tracking.

    This function sets up wandb logging with the experiment configuration and name.
    It enables tracking of metrics, hyperparameters, and model artifacts during training.

    Args:
        cfg: Hydra configuration object containing logging settings
            - logging.enable_wandb: Whether to enable wandb logging
            - logging.experiment_name: Name of the experiment
            - logging.project_name: Name of the wandb project
    """
    if cfg.logging.enable_wandb:
        # Use the experiment name from logging config
        run_name = cfg.logging.experiment_name

        wandb.init(
            project=cfg.logging.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
        )


def create_model_and_dataset(cfg: DictConfig) -> Tuple[tf.keras.Model, Any]:
    """Create and initialize the model and dataset based on configuration.

    This function instantiates the model and dataset using Hydra's instantiation
    system, configures the loss function, and compiles the model with the
    appropriate optimizer and loss.

    Args:
        cfg: Hydra configuration object containing:
            - dataset: Dataset configuration
            - model: Model configuration including loss settings
            - train.learning_rate: Learning rate for optimizer

    Returns:
        Tuple containing:
            - model: Compiled TensorFlow model
            - dataset: Dataset instance for training/validation

    Note:
        The model is compiled with Adam optimizer and the specified loss function
        from the configuration.
    """
    # For training, load train/val data
    dataset = hydra.utils.instantiate(cfg.dataset, mode="train")

    # Pass processed_news to the model
    model = hydra.utils.instantiate(cfg.model, processed_news=dataset.processed_news)

    # Get loss configuration from model
    loss_config = cfg.model.loss
    loss_name = loss_config.name
    loss_kwargs = {k: v for k, v in loss_config.items() if k != "name"}

    # Create loss function using our custom loss system
    loss_fn = get_loss(loss_name, **loss_kwargs)

    # Compile the model with optimizer and loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate),
        loss=loss_fn,
    )

    return model, dataset


def get_num_batches(dataset_size: int, batch_size: int) -> int:
    """Calculate the number of batches per epoch.

    Args:
        dataset_size: Total number of samples in the dataset
        batch_size: Number of samples per batch

    Returns:
        Number of batches needed to process the entire dataset
    """
    return (dataset_size + batch_size - 1) // batch_size


def save_predictions(
    predictions: Dict[str, Any], output_path: Path, epoch: Optional[int] = None, mode: str = "val"
) -> None:
    """Save model predictions and ground truth to a tab-separated file.

    This function creates a structured file containing impression IDs, ground truth labels,
    and model predictions. The file is saved in a tab-separated format for easy analysis
    and visualization.

    Args:
        predictions: Dictionary mapping impression IDs to tuples of (ground_truth, prediction)
        output_path: Directory path where prediction files will be saved
        epoch: Current epoch number (for validation) or None (for testing)
        mode: Either 'val' for validation or 'test' for testing predictions

    Example:
        >>> predictions = {"123": (1.0, 0.85), "124": (0.0, 0.12)}
        >>> save_predictions(predictions, Path("output"), epoch=1, mode="val")
        # Creates: output/val_predictions_epoch_1.txt
        # File contents:
        # ImpressionID    GroundTruth    Prediction
        # 123            1.0            0.85
        # 124            0.0            0.12
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Create filename based on mode and epoch
    filename = f"{mode}_predictions"
    if epoch is not None:
        filename += f"_epoch_{epoch}"
    filename += ".txt"

    filepath = output_path / filename

    with open(filepath, "w") as f:
        f.write("ImpressionID\tGroundTruth\tPrediction\n")
        for imp_id, (gt, pred) in predictions.items():
            f.write(f"{imp_id}\t{gt}\t{pred}\n")

    console.log(f"Saved {mode} predictions to {filepath}")


def validate(
    model,
    dataloader,
    metrics,
    num_impressions,
    progress,
    mode="validate",
    save_predictions_path: Optional[Path] = None,
    epoch: Optional[int] = None,
):
    """Run validation or testing using impression-by-impression processing.

    This function evaluates the model on validation or test data, computing various
    metrics (AUC, MRR, NDCG) and optionally saving predictions for later analysis.
    It processes each impression individually to ensure proper evaluation of the
    recommendation system.

    Args:
        model: The trained news recommendation model
        dataloader: DataLoader providing validation/test data
        metrics: NewsRecommenderMetrics instance for computing evaluation metrics
        num_impressions: Number of impressions to process
        progress: Rich Progress instance for displaying progress bars
        mode: Evaluation mode ('validate' or 'test')
        save_predictions_path: Optional path to save predictions
        epoch: Current epoch number (for validation) or None (for testing)

    Returns:
        Dictionary containing:
            - loss: Average loss across all impressions
            - auc: Average AUC score
            - mrr: Average Mean Reciprocal Rank
            - ndcg@5: Average NDCG@5 score
            - ndcg@10: Average NDCG@10 score
            - num_impressions: Number of processed impressions

    Note:
        If save_predictions_path is provided, predictions will be saved in a
        tab-separated file with impression IDs, ground truth, and predictions.
    """
    # Create progress bar for validation
    val_progress = progress.add_task(f"Running {mode}...", total=num_impressions, visible=True)

    # Initialize metrics
    val_loss = 0
    metric_values = {"auc": [], "mrr": [], "ndcg@5": [], "ndcg@10": []}
    num_processed = 0

    # Store predictions if needed
    predictions = {}

    # Process each impression
    for features, labels, impression_ids in dataloader:
        if num_processed >= num_impressions:
            break

        # Use model's test_step for validation
        step_metrics = model.test_step((features, labels))

        # Extract loss and update metrics
        val_loss += float(step_metrics["loss"])

        # Calculate additional metrics for this impression
        scores = model(features, training=False)
        impression_metrics = metrics.compute_metrics(labels, scores, progress=None)

        # Store predictions if needed
        if save_predictions_path is not None:
            for imp_id, gt, pred in zip(impression_ids.numpy(), labels.numpy(), scores.numpy()):
                # Handle arrays by taking the first element if needed
                gt_value = float(gt[0]) if gt.ndim > 0 else float(gt)
                pred_value = float(pred[0]) if pred.ndim > 0 else float(pred)
                predictions[str(imp_id)] = (gt_value, pred_value)

        # Store metrics for this impression
        for metric_name, value in impression_metrics.items():
            metric_values[metric_name].append(float(value))

        # Update progress
        progress.update(val_progress, advance=1)
        num_processed += 1

    # Calculate average metrics across all impressions
    final_metrics = {"loss": val_loss / num_processed}

    # Average all collected metrics
    for metric_name, values in metric_values.items():
        # Only compute if we have values
        final_metrics[metric_name] = sum(values) / len(values) if values else 0.0

    final_metrics["num_impressions"] = num_processed

    # Save predictions if needed
    if save_predictions_path is not None:
        save_predictions(predictions, save_predictions_path, epoch, mode)

    return final_metrics


def log_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Log metrics to the console with optional prefix.
    
    Args:
        metrics: Dictionary of metric names and their values
        prefix: Optional prefix to add before each metric name
    """
    if prefix:
        console.log(f"[bold blue]{prefix}")
    for metric_name, value in metrics.items():
        console.log(f"[bold blue]{metric_name}: {value:.4f}")


def save_metrics_to_file(
    metrics_path: Path,
    initial_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    wandb_history: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Save training metrics to a JSON file.

    This function saves all relevant metrics (initial, best validation, and test)
    along with training history to a JSON file for later analysis.

    Args:
        metrics_path: Path where the metrics file will be saved
        initial_metrics: Metrics from initial validation
        best_metrics: Best validation metrics achieved
        test_metrics: Final test metrics
        wandb_history: Optional training history from wandb
    """
    metrics_data = {
        "initial": {k: float(v) for k, v in initial_metrics.items()},
        "best_validation": {k: float(v) for k, v in best_metrics.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
    }
    if wandb_history:
        metrics_data["training_history"] = wandb_history

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    console.log(f"Saved metrics to {metrics_path}")


def calculate_validation_metrics(
    val_metrics: Dict[str, float],
) -> Tuple[float, float, float, float, float]:
    """Calculate validation metrics and their average.

    Args:
        val_metrics: Dictionary containing validation metrics

    Returns:
        Tuple containing:
            - auc: Area Under Curve score
            - mrr: Mean Reciprocal Rank
            - ndcg@5: Normalized Discounted Cumulative Gain at 5
            - ndcg@10: Normalized Discounted Cumulative Gain at 10
            - avg_metric: Average of all metrics
    """
    current_val_auc = float(val_metrics.get("auc", 0.0))
    current_val_mrr = float(val_metrics.get("mrr", 0.0))
    current_val_ndcg5 = float(val_metrics.get("ndcg@5", 0.0))
    current_val_ndcg10 = float(val_metrics.get("ndcg@10", 0.0))
    current_val_avg_metric = (
        current_val_auc + current_val_mrr + current_val_ndcg5 + current_val_ndcg10
    ) / 4.0
    return (
        current_val_auc,
        current_val_mrr,
        current_val_ndcg5,
        current_val_ndcg10,
        current_val_avg_metric,
    )


def update_best_metrics(
    best_metrics: Dict[str, Any],
    epoch: int,
    epoch_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    current_val_avg_metric: float,
) -> bool:
    """Update best metrics if validation metric improved.

    Args:
        best_metrics: Dictionary storing the best metrics so far
        epoch: Current epoch number
        epoch_metrics: Metrics from current epoch
        val_metrics: Validation metrics from current epoch
        current_val_avg_metric: Average validation metric for current epoch

    Returns:
        True if metrics improved, False otherwise
    """
    if current_val_avg_metric > best_metrics["val_avg_metric"]:
        best_metrics |= {
            "epoch": epoch,
            "loss": float(epoch_metrics["train/loss"]),
            "auc": float(val_metrics.get("auc", 0.0)),
            "mrr": float(val_metrics.get("mrr", 0.0)),
            "ndcg@5": float(val_metrics.get("ndcg@5", 0.0)),
            "ndcg@10": float(val_metrics.get("ndcg@10", 0.0)),
            "val_avg_metric": current_val_avg_metric,
        }
        return True
    return False


def log_wandb_metrics(
    wandb_metrics: Dict[str, float], wandb_history: Dict[str, List[float]]
) -> None:
    """Log metrics to wandb and update history.

    Args:
        wandb_metrics: Dictionary of metrics to log
        wandb_history: Dictionary storing metric history
    """
    wandb.log(wandb_metrics)
    for k, v in wandb_metrics.items():
        if k in wandb_history:
            try:
                wandb_history[k].append(float(v))
            except (ValueError, TypeError) as e:
                console.log(f"Could not convert metric {k} with value {v} to float: {e}")


def train_epoch(
    model: tf.keras.Model,
    dataloader: Any,
    num_batches: int,
    progress: Progress,
    log_every_n_steps: int,
    epoch: int,
) -> Dict[str, float]:
    """Train the model for one epoch.

    Args:
        model: The model to train
        dataloader: DataLoader providing training data
        num_batches: Number of batches to process
        progress: Rich Progress instance for displaying progress
        log_every_n_steps: How often to log training progress
        epoch: Current epoch number

    Returns:
        Dictionary containing training metrics for the epoch
    """
    epoch_metrics = {"train/loss": 0.0, "train/batch_losses": []}
    train_progress = progress.add_task(f"Epoch {epoch + 1}", total=num_batches)

    for batch_idx, batch in enumerate(dataloader, 1):
        step_metrics = model.train_step(batch)
        loss = float(step_metrics["loss"])
        epoch_metrics["train/batch_losses"].append(loss)
        epoch_metrics["train/loss"] += loss
        progress.update(train_progress, advance=1)

        if batch_idx % log_every_n_steps == 0:
            console.log(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")

    epoch_metrics["train/loss"] /= num_batches
    return epoch_metrics


def log_epoch_metrics(
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    wandb_history: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Log metrics for an epoch to console and wandb.

    Args:
        epoch: Current epoch number
        train_metrics: Training metrics for the epoch
        val_metrics: Validation metrics for the epoch
        wandb_history: Optional wandb history for logging
    """
    console.log(f"[bold cyan]Epoch {epoch}")
    console.log(f"[bold green]Training Loss: {train_metrics['train/loss']:.4f}")
    log_metrics(val_metrics, "Validation Metrics:\n")

    if wandb_history is not None:
        wandb_metrics = {
            "epoch": epoch,
            "train/loss": train_metrics["train/loss"],
            **{
                f"val/{k}": v
                for k, v in val_metrics.items()
                if k in ["loss", "auc", "mrr", "ndcg@5", "ndcg@10"]
            },
        }
        log_wandb_metrics(wandb_metrics, wandb_history)


def run_training_loop(
    model: tf.keras.Model,
    dataset: Any,
    cfg: DictConfig,
    metrics: NewsRecommenderMetrics,
    progress: Progress,
    best_metrics: Dict[str, Any],
    wandb_history: Optional[Dict[str, List[float]]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run the main training loop with validation and early stopping.

    Args:
        model: The model to train
        dataset: Dataset instance providing training and validation data
        cfg: Configuration object containing training settings
        metrics: NewsRecommenderMetrics instance for evaluation
        progress: Rich Progress instance for displaying progress
        best_metrics: Dictionary to store best validation metrics
        wandb_history: Optional wandb history for logging

    Returns:
        Tuple containing:
            - train_metrics: Final training metrics
            - val_metrics: Final validation metrics
    """
    num_train_batches = get_num_batches(dataset.train_size, cfg.train.batch_size)
    num_val_batches = get_num_batches(dataset.val_size, cfg.train.batch_size)
    patience_counter = 0
    best_model_path = Path(cfg.train.model_dir) / "best_epoch.weights.h5"
    last_model_path = Path(cfg.train.model_dir) / "last_epoch.weights.h5"
    predictions_dir = Path(cfg.train.model_dir) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    epoch_progress = progress.add_task(
        f"Training for {cfg.train.num_epochs} epochs", total=cfg.train.num_epochs
    )

    for epoch in range(cfg.train.num_epochs):
        # Training
        train_metrics = train_epoch(
            model,
            dataset.train_dataloader(cfg.train.batch_size),
            num_train_batches,
            progress,
            cfg.logging.log_every_n_steps,
            epoch,
        )

        # Validation
        val_metrics = validate(
            model,
            dataset.val_dataloader(),
            metrics,
            num_val_batches,
            progress,
            mode="validate",
            save_predictions_path=predictions_dir,
            epoch=epoch,
        )

        # Log metrics
        log_epoch_metrics(
            epoch,
            train_metrics,
            val_metrics,
            wandb_history if cfg.logging.enable_wandb else None,
        )

        # Save model
        model.save_weights(last_model_path)

        # Update best metrics
        current_val_metrics = calculate_validation_metrics(val_metrics)
        if update_best_metrics(
            best_metrics,
            epoch,
            train_metrics,
            val_metrics,
            current_val_metrics[-1],
        ):
            patience_counter = 0
            model.save_weights(best_model_path)
            console.log(
                f"Saved best model at epoch {epoch} with avg val_metric: "
                f"{current_val_metrics[-1]:.4f} (mrr: {current_val_metrics[1]:.4f}, "
                f"train loss: {train_metrics['train/loss']:.4f})"
            )
        else:
            patience_counter += 1

        if patience_counter >= cfg.train.early_stopping_patience:
            console.log("Early stopping triggered")
            break

        progress.update(epoch_progress, advance=1)

    return train_metrics, val_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function for the news recommendation model.
    
    This function orchestrates the entire training process:
    1. Sets up device and logging
    2. Creates model and dataset
    3. Runs initial validation
    4. Trains the model with validation and early stopping
    5. Tests the model with the best weights
    6. Saves all metrics and predictions
    
    Args:
        cfg: Hydra configuration object containing all training settings:
            - device: GPU and memory settings
            - train: Training hyperparameters
            - logging: Logging and wandb settings
            - model: Model architecture and loss settings
            - dataset: Dataset configuration
    """
    console.log("🚀 Starting training...")
    setup_device(
        gpu_ids=cfg.device.gpu_ids,
        memory_limit=cfg.device.memory_limit,
        mixed_precision=cfg.device.mixed_precision,
    )

    # Setup model and dataset
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.train.model_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log save locations
    console.log("\n[bold yellow]Save Locations:")
    console.log(f"📁 Run directory: {run_dir}")
    console.log(f"💾 Model weights: {run_dir}/best_epoch.weights.h5")
    console.log(f"📊 Metrics: {run_dir}/metrics.json")
    console.log(f"📈 Predictions: {run_dir}/predictions/\n")

    tf.random.set_seed(cfg.seed)
    model, dataset = create_model_and_dataset(cfg)
    setup_wandb(cfg)

    # Initialize metrics
    metrics = NewsRecommenderMetrics()
    best_metrics = {
        "epoch": 0,
        "loss": float("inf"),
        "auc": 0.0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
        "val_avg_metric": 0.0,
    }

    wandb_history = {
        "epoch": [],
        "train/loss": [],
        "val/loss": [],
        "val/auc": [],
        "val/mrr": [],
        "val/ndcg@5": [],
        "val/ndcg@10": [],
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total} samples left"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        # Initial validation
        console.log("\n[bold yellow]Running initial validation...")
        initial_metrics = validate(
            model,
            dataset.val_dataloader(),
            metrics,
            get_num_batches(dataset.val_size, cfg.train.batch_size),
            progress,
            mode="initial_validation",
            save_predictions_path=run_dir / "predictions",
            epoch=0,
        )
        log_metrics(initial_metrics, "Initial Metrics:")

        # Run training loop
        _, _ = run_training_loop(
            model,
            dataset,
            cfg,
            metrics,
            progress,
            best_metrics,
            wandb_history if cfg.logging.enable_wandb else None,
        )

        # Testing phase
        if Path(cfg.train.model_dir) / "best_epoch.weights.h5".exists():
            model.load_weights(Path(cfg.train.model_dir) / "best_epoch.weights.h5")
            console.log(f"Loaded best model from epoch {best_metrics['epoch']}")

            console.log("\n[bold yellow]Loading test data...")
            dataset._load_data("test")

            console.log("\n[bold yellow]Starting Testing Phase...")
            test_metrics = validate(
                model,
                dataset.test_dataloader(),
                metrics,
                get_num_batches(dataset.test_size, cfg.train.batch_size),
                progress,
                mode="test",
                save_predictions_path=run_dir / "predictions",
                epoch=best_metrics["epoch"],
            )

            if cfg.logging.enable_wandb:
                wandb.log(
                    {
                        "test/final": {
                            **{f"test/{k}": v for k, v in test_metrics.items()},
                            "best_epoch": best_metrics["epoch"],
                        }
                    }
                )
                wandb.run.summary.update(
                    {
                        "best_val/epoch": best_metrics["epoch"],
                        "best_val/avg_metric": best_metrics["val_avg_metric"],
                        **{
                            f"best_val/{k}": v
                            for k, v in best_metrics.items()
                            if k not in ["epoch", "val_avg_metric", "loss"]
                        },
                        "best_val/train_loss_at_best_avg": best_metrics["loss"],
                    }
                )

            log_metrics(test_metrics, "Test Metrics:\n")
            save_metrics_to_file(
                run_dir / "metrics.json",
                initial_metrics,
                best_metrics,
                test_metrics,
                wandb_history if cfg.logging.enable_wandb else None,
            )

    if cfg.logging.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()

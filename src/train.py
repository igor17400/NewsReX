import os

# Set TensorFlow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import datetime
import numpy as np

from pathlib import Path
from typing import Any, Dict, Tuple

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

# Setup logging
setup_logging()

console = Console()
rich_handler = RichHandler(console=console, rich_tracebacks=True, show_time=True)


def setup_wandb(cfg: DictConfig) -> None:
    """Initialize Weights & Biases logging"""
    if cfg.logging.enable_wandb:
        # Use the experiment name from logging config
        run_name = cfg.logging.experiment_name

        wandb.init(
            project=cfg.logging.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
        )


def create_model_and_dataset(cfg: DictConfig) -> Tuple[tf.keras.Model, Any]:
    """Create model and dataset instances based on config"""
    # For training, load train/val data
    dataset = hydra.utils.instantiate(cfg.dataset, mode="train")

    # Pass processed_news to the model
    model = hydra.utils.instantiate(cfg.model, processed_news=dataset.processed_news)

    return model, dataset


def train_step(model, batch, optimizer):
    """Perform a single training step."""
    features, labels = batch

    with tf.GradientTape() as tape:
        # Forward pass (will apply softmax internally)
        scores = model(features, training=True)

        # Cast labels to match scores dtype
        labels = tf.cast(labels, scores.dtype)

        # Calculate categorical crossentropy loss
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                labels, scores, from_logits=False  # scores already have softmax applied
            )
        )

        # Scale loss for mixed precision
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            loss = optimizer.get_scaled_loss(loss)

    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Unscale gradients if using mixed precision
    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        gradients = optimizer.get_unscaled_gradients(gradients)

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def get_num_batches(dataset_size: int, batch_size: int) -> int:
    """Calculate number of batches per epoch"""
    return (dataset_size + batch_size - 1) // batch_size


def validate(model, dataloader, metrics, num_impressions, progress, mode="validate"):
    """Run validation/testing using impression-by-impression processing."""
    # Create progress bar for validation
    val_progress = progress.add_task(f"Running {mode}...", total=num_impressions, visible=True)

    # Initialize metrics
    val_loss = 0
    metric_values = {
        "auc": [],
        "mrr": [],
        "ndcg@5": [],
        "ndcg@10": [],
        "group_auc": []
    }
    num_processed = 0

    # Process each impression
    for features, labels, impression_ids in dataloader:
        if num_processed >= num_impressions:
            break
            
        # Get predictions (will apply sigmoid internally)
        scores = model(features, training=False)
        
        # Ensure consistent dtype (float32) for metrics computation
        scores = tf.cast(scores, tf.float32)
        labels = tf.cast(labels, tf.float32)
        
        # Calculate loss using categorical crossentropy
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                labels,
                scores,
                from_logits=False  # scores already have activation applied
            )
        )
        
        val_loss += float(loss)

        # Calculate metrics for this impression
        impression_metrics = metrics.compute_metrics(
            labels,
            scores,
            progress=None
        )
        
        # Store metrics for this impression
        for metric_name, value in impression_metrics.items():
            metric_values[metric_name].append(float(value))

        # Update progress
        progress.update(val_progress, advance=1)
        num_processed += 1

        # Clear memory after each impression
        tf.keras.backend.clear_session()

    try:
        # Calculate average metrics across all impressions
        final_metrics = {
            "loss": val_loss / num_processed
        }
        
        # Average all collected metrics
        for metric_name, values in metric_values.items():
            if values:  # Only compute if we have values
                final_metrics[metric_name] = sum(values) / len(values)
            else:
                final_metrics[metric_name] = 0.0

        final_metrics["num_impressions"] = num_processed
        return final_metrics

    finally:
        # Clear memory
        tf.keras.backend.clear_session()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training loop."""
    console.log("🚀 Starting training...")
    console.log("⚙️ Configuring devices...")

    # Setup devices first
    setup_device(
        gpu_ids=cfg.device.gpu_ids,
        memory_limit=cfg.device.memory_limit,
        mixed_precision=cfg.device.mixed_precision,
    )

    # Create model directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(cfg.train.model_dir)
    run_dir = model_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"Model checkpoints will be saved to: {run_dir}")

    # Set random seeds
    tf.random.set_seed(cfg.seed)

    # Create model and dataset first
    model, dataset = create_model_and_dataset(cfg)

    # Setup wandb after dataset is loaded (so we have all the info)
    setup_wandb(cfg)
    console.log(f"📋 Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)

    # Create metrics
    metrics = NewsRecommenderMetrics()

    # Calculate number of batches using new size properties
    num_train_batches = get_num_batches(dataset.train_size, cfg.train.batch_size)
    num_val_batches = get_num_batches(dataset.val_size, cfg.train.batch_size)

    console.log(f"Number of train batches: {num_train_batches}")
    console.log(f"Number of validation batches: {num_val_batches}")

    # At the beginning of train function, initialize best_metrics with Python types
    best_metrics = {
        "epoch": 0,
        "loss": float("inf"),  # Initialize with infinity
        "auc": 0.0,
        "group_auc": 0.0,  # Add group_auc
        "mrr": 0.0,
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
    }

    # Training loop
    patience_counter = 0

    best_model_path = run_dir / "best_epoch.weights.h5"
    last_model_path = run_dir / "last_epoch.weights.h5"

    # Initialize wandb metrics history
    wandb_history = {
        "epoch": [],
        "train/loss": [],
        "val/loss": [],
        "val/auc": [],
        "val/group_auc": [],
        "val/mrr": [],
        "val/ndcg@5": [],
        "val/ndcg@10": []
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
        # Run initial validation to get baseline performance
        console.log("\n[bold yellow]Running initial validation (before training)...")
        initial_metrics = validate(
            model,
            dataset.val_dataloader(),
            metrics,
            num_val_batches,
            progress,
            mode="initial validation",
        )

        # Log initial metrics
        console.log("\n[bold yellow]Initial Metrics (before training):")
        for metric_name, value in initial_metrics.items():
            console.log(f"[bold blue]{metric_name}: {value:.4f}")

        epoch_progress = progress.add_task(
            f"Training for {cfg.train.num_epochs} epochs", total=cfg.train.num_epochs
        )

        for epoch in range(cfg.train.num_epochs):
            # Training
            train_progress = progress.add_task(
                f"Epoch {epoch + 1}/{cfg.train.num_epochs}", total=num_train_batches
            )

            # Training metrics for the epoch
            epoch_metrics = {
                "train/loss": 0.0,
                "train/batch_losses": [],
            }

            # Initialize num_batches for this epoch
            num_batches = 0

            for batch in dataset.train_dataloader(cfg.train.batch_size):
                loss = train_step(model, batch, optimizer)

                # Track batch metrics
                epoch_metrics["train/batch_losses"].append(float(loss))
                epoch_metrics["train/loss"] += float(loss)
                num_batches += 1

                # Update batch progress
                progress.update(train_progress, advance=1)

                if num_batches % cfg.logging.log_every_n_steps == 0:
                    console.log(
                        f"Epoch {epoch}, Batch {num_batches}/{num_train_batches}, Loss: {loss:.4f}"
                    )

            # Calculate average training loss for the epoch
            epoch_metrics["train/loss"] /= num_batches

            # Validation
            val_metrics = validate(
                model,
                dataset.val_dataloader(),
                metrics,
                num_val_batches,
                progress,
                mode="validate",
            )

            # Only prepare and log wandb metrics if enabled
            if cfg.logging.enable_wandb:
                # Prepare metrics dictionary
                wandb_metrics = {
                    "epoch": epoch,
                    "train/loss": epoch_metrics["train/loss"]
                }
                
                # Add validation metrics with proper prefixes
                for metric_name, value in val_metrics.items():
                    if metric_name in ["loss", "auc", "group_auc", "mrr", "ndcg@5", "ndcg@10"]:
                        wandb_metrics[f"val/{metric_name}"] = value
                
                # Log to wandb
                wandb.log(wandb_metrics)

                # Update history
                for k, v in wandb_metrics.items():
                    if k in wandb_history:
                        try:
                            wandb_history[k].append(float(v))
                        except (ValueError, TypeError) as e:
                            console.log(f"Could not convert metric {k} with value {v} to float: {e}")

            # Save last model after each epoch
            model.save_weights(last_model_path)

            # Update best model if loss improved
            if epoch_metrics["train/loss"] < best_metrics["loss"]:
                best_metrics = {
                    "epoch": epoch,
                    "loss": float(epoch_metrics["train/loss"]),
                    **{k: float(v) for k, v in val_metrics.items()},
                }
                patience_counter = 0

                # Save best model
                model.save_weights(best_model_path)
                console.log(
                    f"Saved best model at epoch {epoch} with loss: {epoch_metrics['train/loss']:.4f}"
                )
            else:
                patience_counter += 1

            # Log to console (always do this)
            console.log(f"[bold cyan]Epoch {epoch}")
            console.log(f"[bold green]Training Loss: {epoch_metrics['train/loss']:.4f}")
            console.log("[bold yellow]Validation Metrics:")
            for metric_name, value in val_metrics.items():
                console.log(f"[bold blue]{metric_name}: {value:.4f}")

            # Early stopping check
            if patience_counter >= cfg.train.early_stopping_patience:
                console.log("Early stopping triggered")
                break

            progress.update(epoch_progress, advance=1)

        # Testing phase
        if best_model_path.exists():
            model.load_weights(best_model_path)
            console.log(f"Loaded best model from epoch {best_metrics['epoch']}")

            # Load and process test data
            console.log("\n[bold yellow]Loading test data...")
            dataset._load_data("test")  # Only process test data when needed

            # Run testing
            console.log("\n[bold yellow]Starting Testing Phase...")
            test_metrics = validate(
                model,
                dataset.test_dataloader(),
                metrics,
                get_num_batches(dataset.test_size, cfg.train.batch_size),
                progress,
                mode="test",
            )

            # Only log to wandb if enabled
            if cfg.logging.enable_wandb:
                # Log test metrics
                wandb.log(
                    {
                        "test/final": {
                            **{f"test/{k}": v for k, v in test_metrics.items()},
                            "best_epoch": best_metrics["epoch"],
                        }
                    }
                )

                # Log summary of best validation metrics
                wandb.run.summary.update(
                    {
                        "best_val/epoch": best_metrics["epoch"],
                        **{f"best_val/{k}": v for k, v in best_metrics.items() if k != "epoch"},
                    }
                )

            # Always log to console
            console.log("\n[bold yellow]Test Metrics:")
            for metric_name, value in test_metrics.items():
                console.log(f"[bold blue]{metric_name}: {value:.4f}")

            # Save metrics to the run directory
            metrics_path = run_dir / "metrics.json"
            metrics_data = {
                "initial": {k: float(v) for k, v in initial_metrics.items()},
                "best_validation": {k: float(v) for k, v in best_metrics.items()},
                "test": {k: float(v) for k, v in test_metrics.items()},
            }
            if cfg.logging.enable_wandb:
                metrics_data["training_history"] = wandb_history

            with open(metrics_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
            console.log(f"Saved metrics to {metrics_path}")

    # Only finish wandb if it was enabled
    if cfg.logging.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()

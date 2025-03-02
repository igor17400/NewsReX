import os

# Set TensorFlow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import json
import datetime

from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

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
        wandb.init(
            project=cfg.logging.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.experiment_name if hasattr(cfg, "experiment_name") else None,
        )


def create_model_and_dataset(cfg: DictConfig) -> Tuple[tf.keras.Model, Any]:
    """Create model and dataset instances based on config"""
    # For training, load train/val data
    dataset = hydra.utils.instantiate(cfg.dataset, mode="train")
    model = hydra.utils.instantiate(cfg.model)
    return model, dataset


def train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    inputs: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]],
    labels: tf.Tensor,
    masks: tf.Tensor,
) -> Tuple[float, tf.Tensor]:
    """Single training step"""
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)  # [batch_size, num_impressions]

        # Normalize labels to create probability distribution
        label_probs = labels / tf.reduce_sum(
            labels * tf.cast(masks, tf.float32), axis=1, keepdims=True
        )

        # Calculate categorical cross entropy
        rank_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        losses = rank_loss_fn(label_probs, predictions)  # [batch_size]

        # Apply mask and compute mean
        loss = tf.reduce_mean(losses)

    # Compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, predictions


def get_num_batches(dataset_size: int, batch_size: int) -> int:
    """Calculate number of batches per epoch"""
    return (dataset_size + batch_size - 1) // batch_size


def validate(
    model: tf.keras.Model,
    dataloader: Iterator,
    metrics: NewsRecommenderMetrics,
    num_batches: int,
    progress: Progress,
    mode: str = "validate",
) -> Dict[str, float]:
    """Run validation or testing and compute metrics"""
    all_labels = []
    all_predictions = []
    all_masks = []
    all_impression_ids = []

    # Set task description based on mode
    task_description = "[cyan]Validating" if mode == "validate" else "[cyan]Testing"
    task = progress.add_task(task_description, total=num_batches)

    batch_idx = 0
    for inputs, labels, masks in dataloader:
        predictions = model(inputs, training=False)
        all_labels.append(labels)
        all_predictions.append(predictions)
        all_masks.append(masks)
        batch_size = labels.shape[0]
        batch_impression_ids = [f"imp_{batch_idx}_{i}" for i in range(batch_size)]
        all_impression_ids.extend(batch_impression_ids)
        batch_idx += 1
        progress.advance(task)

    labels = tf.concat(all_labels, axis=0)
    predictions = tf.concat(all_predictions, axis=0)
    masks = tf.concat(all_masks, axis=0)

    # Get metrics and convert tensors to Python types
    metric_values = metrics.group_metrics(labels, predictions, all_impression_ids, masks=masks)
    return {k: float(v) for k, v in metric_values.items()}  # Convert tensors to Python floats


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

    # Setup
    setup_wandb(cfg)
    console.log(f"📋 Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create model directory if it doesn't exist
    model_dir = Path(cfg.train.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"Model checkpoints will be saved to: {model_dir}")

    # Set random seeds
    tf.random.set_seed(cfg.seed)

    # Create model and dataset
    model, dataset = create_model_and_dataset(cfg)

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)

    # Create metrics
    metrics = NewsRecommenderMetrics()

    # Calculate number of batches
    num_train_batches = get_num_batches(dataset.n_train_behaviors, cfg.train.batch_size)
    console.log(f"Number of train batches: {num_train_batches}")
    num_val_batches = get_num_batches(dataset.n_val_behaviors, cfg.train.batch_size)
    console.log(f"Number of validation batches: {num_val_batches}")

    # At the beginning of train function, initialize best_metrics with Python types
    best_metrics = {
        "epoch": 0,
        "loss": float("inf"),  # Initialize with infinity
        "auc": 0.0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
    }

    # Training loop
    patience_counter = 0

    best_checkpoint_time = None

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
        epochs_task = progress.add_task(
            f"[green]Training for {cfg.train.num_epochs} epochs", total=cfg.train.num_epochs
        )

        for epoch in range(cfg.train.num_epochs):
            total_loss: float = 0.0
            num_batches: int = 0

            # Batch progress with total count
            batch_task = progress.add_task(
                f"[yellow]Epoch {epoch+1}/{cfg.train.num_epochs}", total=num_train_batches
            )

            for batch in dataset.train_dataloader(cfg.train.batch_size):
                inputs, labels, masks = batch
                loss, _ = train_step(model, optimizer, inputs, labels, masks)

                total_loss += loss
                num_batches += 1

                # Update batch progress
                progress.update(batch_task, advance=1)

                if num_batches % cfg.logging.log_every_n_steps == 0:
                    console.log(
                        f"Epoch {epoch}, Batch {num_batches}/{num_train_batches}, Loss: {loss:.4f}"
                    )
                    if cfg.logging.enable_wandb:
                        wandb.log({"train/loss": loss, "train/step": epoch * num_batches})

            avg_loss = total_loss / num_batches

            # Validation
            val_metrics = validate(
                model,
                dataset.val_dataloader(cfg.train.batch_size),
                metrics,
                num_val_batches,
                progress,
                mode="validate",
            )

            # Update best metrics only if loss improved
            if avg_loss < best_metrics["loss"]:
                best_metrics = {
                    "epoch": epoch,
                    "loss": float(avg_loss),  # Convert to Python float
                    **{k: float(v) for k, v in val_metrics.items()}  # Convert all metrics to Python floats
                }
                patience_counter = 0

                best_checkpoint_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                # Save best model
                model_path = (
                    model_dir / f"{cfg.model._target_}_best_{best_checkpoint_time}.weights.h5"
                )
                model.save_weights(model_path)
                console.log(f"Saved best model at epoch {epoch} with loss: {avg_loss:.4f}")
            else:
                patience_counter += 1

            # Log current metrics
            console.log(f"Epoch {epoch} - Loss: {avg_loss:.4f}, AUC: {val_metrics['auc']:.4f}")

            if cfg.logging.enable_wandb:
                wandb.log(
                    {
                        "val/loss": avg_loss,
                        **{f"val/{k}": v for k, v in val_metrics.items()},
                        "epoch": epoch,
                    }
                )

            # Early stopping check
            if patience_counter >= cfg.train.early_stopping_patience:
                console.log("Early stopping triggered")
                break

            progress.update(epochs_task, advance=1)

        # Display best validation metrics
        print("\nBest Validation Metrics:")
        print(f"Best Epoch: {best_metrics['epoch']}")
        for metric_name, value in best_metrics.items():
            if metric_name != "epoch":
                print(f"{metric_name}: {value:.4f}")

        if best_checkpoint_time:
            best_model_path = (
                model_dir / f"{cfg.model._target_}_best_{best_checkpoint_time}.weights.h5"
            )
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
                    dataset.test_dataloader(cfg.train.batch_size),
                    metrics,
                    get_num_batches(dataset.n_test_behaviors, cfg.train.batch_size),
                    progress,
                    mode="test",
                )

                # Display test metrics
                print("\nTest Metrics:")
                for metric_name, value in test_metrics.items():
                    print(f"{metric_name}: {value:.4f}")

                # Save all metrics to file
                metrics_path = model_dir / "metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump({
                        "best_validation": best_metrics,
                        "test": test_metrics  # Already converted to Python types in validate()
                    }, f, indent=2)
                console.log(f"Saved metrics to {metrics_path}")

    if cfg.logging.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()

import logging
import os
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

# Set TensorFlow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

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
    # Instantiate dataset
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Instantiate model
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
        label_probs = labels / tf.reduce_sum(labels * tf.cast(masks, tf.float32), axis=1, keepdims=True)

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
    val_dataloader: Iterator,
    metrics: NewsRecommenderMetrics,
    num_val_batches: int,
) -> Dict[str, float]:
    """Run validation and compute metrics"""
    all_labels = []
    all_predictions = []
    all_masks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        validate_task = progress.add_task("[cyan]Validating", total=num_val_batches)

        for inputs, labels, masks in val_dataloader:
            predictions = model(inputs, training=False)
            all_labels.append(labels)
            all_predictions.append(predictions)
            all_masks.append(masks)
            progress.advance(validate_task)

    # Concatenate all batches
    labels = tf.concat(all_labels, axis=0)
    predictions = tf.concat(all_predictions, axis=0)
    masks = tf.concat(all_masks, axis=0)

    # Compute metrics
    metric_values = metrics.group_metrics(labels, predictions, masks=masks)

    return metric_values


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training loop."""
    logger.info("🚀 Starting training...")
    logger.info("⚙️ Configuring devices...")
    print("------- OIIII ----------")
    return

    # Setup devices first
    setup_device(gpu_ids=cfg.device.gpu_ids, memory_limit=cfg.device.memory_limit, mixed_precision=cfg.device.mixed_precision)

    # Setup
    setup_wandb(cfg)
    logger.info(f"📋 Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seeds
    tf.random.set_seed(cfg.seed)

    # Create model and dataset
    model, dataset = create_model_and_dataset(cfg)

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)

    # Create metrics
    metrics = NewsRecommenderMetrics()

    # Calculate number of batches
    num_train_batches = get_num_batches(dataset.n_behaviors, cfg.train.batch_size)
    logger.info(f"Number of train batches: {num_train_batches}.")
    num_val_batches = get_num_batches(dataset.n_val_behaviors, cfg.train.batch_size)
    logger.info(f"Number of validation batches: {num_val_batches}.")

    # Training loop
    best_val_auc: float = 0.0
    patience_counter: int = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        # Main training progress
        epochs_task = progress.add_task(f"[green]Training for {cfg.train.num_epochs} epochs", total=cfg.train.num_epochs)

        for epoch in range(cfg.train.num_epochs):
            total_loss: float = 0.0
            num_batches: int = 0

            # Batch progress with total count
            batch_task = progress.add_task(f"[yellow]Epoch {epoch+1}/{cfg.train.num_epochs}", total=num_train_batches)

            for batch in dataset.train_dataloader(cfg.train.batch_size):
                inputs, labels, masks = batch
                loss, _ = train_step(model, optimizer, inputs, labels, masks)

                total_loss += loss
                num_batches += 1

                # Update batch progress
                progress.update(batch_task, advance=1)

                if num_batches % cfg.logging.log_every_n_steps == 0:
                    logger.info(f"Epoch {epoch}, Batch {num_batches}/{num_train_batches}, Loss: {loss:.4f}")
                    if cfg.logging.enable_wandb:
                        wandb.log({"train/loss": loss, "train/step": epoch * num_batches})

            avg_loss = total_loss / num_batches

            # Validation with batch count
            val_metrics = validate(model, dataset.val_dataloader(cfg.train.batch_size), metrics, num_val_batches)

            # Log metrics
            logger.info(f"Epoch {epoch}")
            logger.info(f"Average Loss: {avg_loss:.4f}")
            logger.info("Validation Metrics:")
            for metric_name, value in val_metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")

            # Early stopping logic
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                patience_counter = 0
                model_path = Path(cfg.train.model_dir) / f"{cfg.model._target_}_best.h5"
                model.save_weights(model_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.train.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

            # Update epoch progress
            progress.update(epochs_task, advance=1)
            # Instead of removing the task, just update its description
            progress.update(batch_task, description=f"[blue]Completed Epoch {epoch+1}")

    if cfg.logging.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()

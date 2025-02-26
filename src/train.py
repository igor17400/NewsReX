import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import hydra
import tensorflow as tf
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils.metrics import NewsRecommenderMetrics

logger = logging.getLogger(__name__)


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
) -> Dict[str, float]:
    """Run validation and compute metrics"""
    all_labels = []
    all_predictions = []
    all_masks = []

    # Progress bar for validation
    for inputs, labels, masks in tqdm(val_dataloader, desc="Validating", leave=False):
        predictions = model(inputs, training=False)
        all_labels.append(labels)
        all_predictions.append(predictions)
        all_masks.append(masks)

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
    # Setup
    setup_wandb(cfg)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seeds
    tf.random.set_seed(cfg.seed)

    # Create model and dataset
    model, dataset = create_model_and_dataset(cfg)

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)

    # Create metrics
    metrics = NewsRecommenderMetrics()

    # Calculate total number of batches per epoch
    num_batches_per_epoch = get_num_batches(dataset.n_behaviors, cfg.train.batch_size)

    # Training loop
    best_val_auc: float = 0.0
    patience_counter: int = 0

    # Progress bar for epochs
    epoch_pbar = tqdm(range(cfg.train.num_epochs), desc="Training epochs")
    for epoch in epoch_pbar:
        total_loss: float = 0.0
        num_batches: int = 0

        # Progress bar for batches
        batch_pbar = tqdm(
            dataset.train_dataloader(cfg.train.batch_size),
            total=num_batches_per_epoch,
            desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}",
            leave=False,
        )
        for batch in batch_pbar:
            inputs, labels, masks = batch
            loss, _ = train_step(model, optimizer, inputs, labels, masks)

            total_loss += loss
            num_batches += 1

            # Update batch progress bar
            batch_pbar.set_postfix({"loss": f"{loss:.4f}"})

            if num_batches % cfg.logging.log_every_n_steps == 0:
                logger.info(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss:.4f}")
                if cfg.logging.enable_wandb:
                    wandb.log({"train/loss": loss, "train/step": epoch * num_batches})

        avg_loss = total_loss / num_batches

        # Validation
        val_metrics = validate(model, dataset.val_dataloader(cfg.train.batch_size), metrics)

        # Update epoch progress bar
        epoch_pbar.set_postfix({"loss": f"{avg_loss:.4f}", "val_auc": f"{val_metrics['auc']:.4f}"})

        # Logging
        logger.info(f"Epoch {epoch}")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info("Validation Metrics:")
        for metric_name, value in val_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

        if cfg.logging.enable_wandb:
            wandb.log(
                {
                    "train/epoch_loss": avg_loss,
                    "epoch": epoch,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }
            )

        # Early stopping
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            # Save best model
            model_path = Path(cfg.train.model_dir) / f"{cfg.model._target_}_best.h5"
            model.save_weights(model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.early_stopping_patience:
                logger.info("Early stopping triggered")
                break

    if cfg.logging.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()

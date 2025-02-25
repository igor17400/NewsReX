import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import tensorflow as tf
import wandb
from omegaconf import DictConfig, OmegaConf

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
    news_input: Dict[str, tf.Tensor],
    history_input: tf.Tensor,
    labels: tf.Tensor,
) -> Tuple[float, tf.Tensor]:
    """Single training step"""
    with tf.GradientTape() as tape:
        predictions = model((news_input, history_input), training=True)
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)

    # Compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, predictions


def validate(
    model: tf.keras.Model, val_data: Tuple, metrics: NewsRecommenderMetrics
) -> Dict[str, float]:
    """Run validation and compute metrics"""
    news_input, history_input, labels, impression_ids = val_data

    predictions = model((news_input, history_input), training=False)

    # Compute metrics
    metric_values = metrics.group_metrics(labels, predictions, impression_ids)

    return metric_values


@hydra.main(config_path="../configs", config_name="config")
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

    # Get training and validation data
    news_input, history_input, labels = dataset.get_train_data()  # Use the values directly
    val_data = dataset.get_validation_data()

    # Create metrics
    metrics = NewsRecommenderMetrics()

    # Training loop
    best_val_auc: float = 0.0
    patience_counter: int = 0

    for epoch in range(cfg.train.num_epochs):
        total_loss: float = 0.0
        num_batches: int = 0

        for batch in dataset.train_dataloader(cfg.train.batch_size):
            news_input, history_input, labels = batch
            loss, predictions = train_step(model, optimizer, news_input, history_input, labels)

            total_loss += loss
            num_batches += 1

            if num_batches % cfg.logging.log_every_n_steps == 0:
                logger.info(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss:.4f}")
                if cfg.logging.enable_wandb:
                    wandb.log({"train/loss": loss, "train/step": epoch * num_batches})

        avg_loss = total_loss / num_batches

        # Validation
        val_metrics = validate(model, val_data, metrics)

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

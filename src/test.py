import logging
from pathlib import Path

import hydra
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
            name=f"test_{cfg.experiment_name}" if hasattr(cfg, "experiment_name") else "test",
        )


@hydra.main(config_path="../configs", config_name="config")
def test(cfg: DictConfig) -> None:
    """Main testing function"""
    # Setup
    setup_wandb(cfg)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create model and dataset
    model = hydra.utils.instantiate(cfg.model)
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Load best model weights
    model_path = Path(cfg.train.model_dir) / f"{cfg.model._target_}_best.h5"
    model.load_weights(model_path)

    # Get test data
    test_data = dataset.get_test_data()

    # Create metrics
    metrics = NewsRecommenderMetrics()

    # Run evaluation
    news_input, history_input, labels, impression_ids = test_data
    predictions = model((news_input, history_input), training=False)

    # Compute metrics
    test_metrics = metrics.group_metrics(labels, predictions, impression_ids)

    # Log results
    logger.info("Test Metrics:")
    for metric_name, value in test_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    if cfg.logging.enable_wandb:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    test()

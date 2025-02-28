import logging
from pathlib import Path

import hydra
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
from utils.metrics import NewsRecommenderMetrics

logger = logging.getLogger(__name__)

# Setup rich logging
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)])


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
    # Setup devices first
    setup_device(gpu_ids=cfg.device.gpu_ids, memory_limit=cfg.device.memory_limit, mixed_precision=cfg.device.mixed_precision)

    # Setup
    setup_wandb(cfg)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create model and dataset
    model = hydra.utils.instantiate(cfg.model)
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Load best model weights
    model_path = Path(cfg.train.model_dir) / f"{cfg.model._target_}_best.h5"
    model.load_weights(model_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        test_task = progress.add_task("[cyan]Loading test data", total=None)

        # Get test data
        test_data = dataset.get_test_data()
        news_input, history_input, labels, masks, impression_ids = test_data

        progress.update(test_task, description="[yellow]Making predictions")
        predictions = model((news_input, history_input), training=False)

        progress.update(test_task, description="[green]Computing metrics")
        metrics = NewsRecommenderMetrics()
        test_metrics = metrics.group_metrics(labels, predictions, impression_ids, masks=masks)

        progress.update(test_task, description="[bold green]Testing completed!", completed=True)

    # Log results
    console = Console()
    console.print("\n[bold green]Test Metrics:[/bold green]")
    for metric_name, value in test_metrics.items():
        console.print(f"{metric_name}: {value:.4f}")

    if cfg.logging.enable_wandb:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    test()

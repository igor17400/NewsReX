from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import wandb

import keras
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress

from .callbacks import FastEvaluationCallback, SlowEvaluationCallback, RichProgressCallback
from src.utils.metrics.functions_optimized import NewsRecommenderMetricsOptimized as NewsRecommenderMetrics
from src.utils.evaluation import (
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

    # JIT compilation warmup to avoid slow first epoch
    try:
        from src.utils.jax_optimizer import warmup_jit_compilation
        console.log("[bold]Warming up JIT compilation to avoid slow first epoch...[/bold]")

        # Get a sample batch for warmup
        model_name = cfg.model._target_.split('.')[-1].lower()
        sample_dataloader = dataset_provider.train_dataloader(
            batch_size=min(2, cfg.train.batch_size),  # Small batch for warmup
            model_name=model_name
        )

        # Get first batch
        for sample_batch in sample_dataloader:
            warmup_jit_compilation(model, sample_batch)
            break

        console.log("[green]JIT warmup completed![/green]")
    except Exception as e:
        console.log(f"[yellow]JIT warmup skipped: {e}[/yellow]")

    initial_metrics = {}
    if cfg.eval.run_initial_eval:
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

    # Model Summary
    if hasattr(model, "training_model") and model.training_model is not None:
        console.log(f"[bold cyan]Summary of {model.name} Training Model (internal):[/bold cyan]")
        model.training_model.summary(print_fn=lambda s: console.log(s))
    if hasattr(model, "scorer_model") and model.scorer_model is not None:
        console.log(f"[bold cyan]Summary of {model.name} Scorer Model (internal):[/bold cyan]")
        model.scorer_model.summary(print_fn=lambda s: console.log(s))

    console.log(f"[bold cyan]Summary of {model.name} Model (main wrapper):[/bold cyan]")
    model.summary(print_fn=lambda s: console.log(s))

    # Prepare training and validation datasets
    console.log("[bold]Preparing datasets for Keras 3 training...[/bold]")

    # Extract model name from config
    model_name = cfg.model._target_.split('.')[-1].lower()  # e.g., "src.models.nrms.NRMS" -> "nrms"

    train_dataset = dataset_provider.train_dataloader(
        batch_size=cfg.train.batch_size,
        model_name=model_name
    )

    # Setup callbacks
    callbacks = []

    # Calculate steps per epoch
    steps_per_epoch = None
    if hasattr(train_dataset, '__len__'):
        try:
            steps_per_epoch = len(train_dataset)
        except:
            steps_per_epoch = None

    # Add Rich progress callback
    progress_callback = RichProgressCallback(progress_bar_manager, cfg.train.num_epochs, steps_per_epoch)
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
        model.fit(
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

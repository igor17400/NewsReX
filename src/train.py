import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Protocol

import hydra
import tensorflow as tf
import wandb
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.metrics import NewsRecommenderMetrics
from utils.io import save_predictions_to_file_fn

from tensorflow.keras import mixed_precision
import numpy as np


# --- Define a Protocol for Dataset (especially for fast_evaluate) ---
# This makes the expected interface for your dataset class explicit.
class FastEvalDataProvider(Protocol):
    train_size: int
    val_size: int
    test_size: int
    processed_news: Dict[str, Any]  # For model __init__

    def train_dataloader(self, batch_size: int, shuffle: bool = True) -> tf.data.Dataset: ...
    def val_dataloader(self, batch_size: int) -> tf.data.Dataset: ...
    def test_dataloader(self, batch_size: int) -> tf.data.Dataset: ...

    # For NRMS.fast_evaluate
    def get_all_news_title_tokens(self) -> np.ndarray: ...  # Shape: (total_news, title_size)
    def get_all_user_history_tokens(
        self,
    ) -> np.ndarray: ...  # Shape: (total_users_in_eval, hist_size, title_size)
    def get_num_impressions(self, mode: str = "val") -> int: ...  # mode can be "val" or "test"

    # Yields: (imp_id:str, user_idx:int, candidate_news_indices:List[int], labels_for_impression:np.ndarray)
    def get_impression_iterator(self, mode: str = "val") -> Any: ...

    # Optional: direct access to pre-structured data for fast_evaluate, if not using iterators
    val_behaviors_data: Dict[str, Any]  # If NRMS.fast_evaluate uses this directly
    test_behaviors_data: Dict[str, Any]


# Initialize Rich Console (can be done once globally)
console = Console()
# setup_logging() # Call this if it initializes your root logger with RichHandler
# If Hydra also configures logging, ensure they don't conflict.


def setup_wandb_session(cfg: DictConfig) -> None:  # Renamed to avoid conflict
    """Initialize Weights & Biases logging."""
    if cfg.logging.enable_wandb:
        run_name = (
            cfg.logging.experiment_name
            or f"nrms_run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        try:
            wandb.init(
                project=cfg.logging.project_name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                name=run_name,
            )
            console.log(
                f"Wandb initialized for project '{cfg.logging.project_name}', run '{run_name}'"
            )
        except Exception as e:
            console.log(f"[red]Failed to initialize Wandb: {e}[/red]")


def train_step_fn(
    model: tf.keras.Model, data: Tuple[Dict[str, tf.Tensor], tf.Tensor]
) -> Dict[str, tf.Tensor]:
    """Custom training step logic. Expects data as (features_dict, labels_tensor)."""
    features, labels = data

    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = model.compute_loss(
            x=features,
            y=labels,
            y_pred=predictions,
            sample_weight=None,
            training=True,
        )

    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    valid_gradients_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
    if not valid_gradients_and_vars and trainable_vars:  # Check if trainable_vars is not empty
        console.log(
            "[yellow][WARNING train_step_fn] No gradients computed for any trainable variables.[/yellow]"
        )
    elif not trainable_vars:
        console.log("[yellow][WARNING train_step_fn] Model has no trainable variables.[/yellow]")
    else:
        model.optimizer.apply_gradients(valid_gradients_and_vars)

    # Update metrics using the recommended approach
    for metric in model.metrics:
        metric.update_state(labels, predictions)

    # Return metrics including loss
    batch_metrics = {m.name: m.result() for m in model.metrics}
    batch_metrics["loss"] = loss  # Current batch loss
    return batch_metrics


def test_step_fn(
    model: tf.keras.Model, data: Tuple[Dict[str, tf.Tensor], tf.Tensor]
) -> Dict[str, tf.Tensor]:
    """Custom test step logic."""
    features, labels = data
    predictions = model(features, training=False)

    loss = model.compiled_loss(labels, predictions, regularization_losses=model.losses)
    model.compiled_metrics.update_state(labels, predictions)

    batch_metrics = {m.name: m.result() for m in model.metrics}
    batch_metrics["loss"] = loss
    return batch_metrics


def initialize_model_and_dataset(cfg: DictConfig) -> Tuple[tf.keras.Model, FastEvalDataProvider]:
    """Instantiate dataset and NRMS model based on Hydra configuration."""
    console.log("Initializing dataset provider...")
    # Dataset instantiation - ensure it matches FastEvalDataProvider protocol if fast_eval is used.
    dataset_provider: FastEvalDataProvider = hydra.utils.instantiate(cfg.dataset, mode="train")

    console.log("Initializing NRMS model...")
    processed_news_data = dataset_provider.processed_news

    nrms_model_params = {
        "word_embeddings_matrix": processed_news_data["embeddings"],
        "vocab_size": processed_news_data["vocab_size"],
        "embedding_dim": cfg.model.embedding_size,
        "history_size": cfg.dataset.max_history_length,
        "title_size": cfg.dataset.max_title_length,
        "num_attention_heads": cfg.model.multiheads,
        "head_dim": cfg.model.head_dim,
        "attention_hidden_dim": cfg.model.attention_hidden_dim,
        "dropout_rate": cfg.model.dropout_rate,
        "seed": cfg.seed,
        "processed_news": processed_news_data,
    }
    # Filter out any None values to allow model defaults if specific params are not in cfg
    nrms_model_params = {k: v for k, v in nrms_model_params.items() if v is not None}

    nrms_model: tf.keras.Model = hydra.utils.instantiate(
        {
            "_target_": cfg.model._target_,
            "processed_news": processed_news_data,
            "embedding_size": cfg.model.embedding_size,
            "multiheads": cfg.model.multiheads,
            "head_dim": cfg.model.head_dim,
            "attention_hidden_dim": cfg.model.attention_hidden_dim,
            "dropout_rate": cfg.model.dropout_rate,
            "seed": cfg.seed,
            # Add any other arguments your NRMS __init__ expects
        },
        _recursive_=False,
    )

    if hasattr(nrms_model, "training_model") and nrms_model.training_model is not None:
        console.log("[bold cyan]Summary of NRMS Training Model (internal):[/bold cyan]")
        nrms_model.training_model.summary(print_fn=lambda s: console.log(s))
    if hasattr(nrms_model, "scorer_model") and nrms_model.scorer_model is not None:
        console.log("[bold cyan]Summary of NRMS Scorer Model (internal):[/bold cyan]")
        nrms_model.scorer_model.summary(print_fn=lambda s: console.log(s))

    console.log("[bold cyan]Summary of NRMS Model (main wrapper):[/bold cyan]")
    nrms_model.summary(print_fn=lambda s: console.log(s))

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)
    if (
        cfg.device.mixed_precision
        and tf.keras.mixed_precision.global_policy().name == "mixed_float16"
    ):
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=False, name="cce_loss")

    compiled_metrics = [tf.keras.metrics.AUC(name="auc")]  # Add more if needed

    nrms_model.compile(optimizer=optimizer, loss=loss_function, metrics=compiled_metrics)
    console.log(
        f"NRMS model compiled. Optimizer: {type(optimizer).__name__}, Loss: {loss_function.name}, Metrics: {[m.name for m in compiled_metrics]}"
    )

    console.log(f"Mixed precision global policy: {tf.keras.mixed_precision.global_policy().name}")
    console.log(f"Optimizer type: {type(optimizer)}")

    return nrms_model, dataset_provider


def calculate_num_batches(dataset_size: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    return (dataset_size + batch_size - 1) // batch_size


def run_evaluation_epoch(
    model: tf.keras.Model,
    eval_dataloader: tf.data.Dataset,
    custom_metrics_calculator: NewsRecommenderMetrics,
    num_total_impressions: int,
    progress: Progress,
    mode: str = "val",  # "val" or "test"
    save_predictions_dir: Optional[Path] = None,
    epoch_idx: Optional[int] = None,  # 0-indexed
):
    """
    Run validation or testing for one epoch (non-fast evaluation).
    This path is used if `cfg.eval.fast_evaluation` is False.
    It iterates through the dataloader, calls `test_step_fn`.
    """
    eval_progress_task = progress.add_task(
        f"Running {mode} (epoch {epoch_idx+1 if epoch_idx is not None else 'N/A'})...",
        total=num_total_impressions,
        visible=True,
    )

    for m in model.metrics:
        m.reset_state()

    all_labels_batched = []
    all_predictions_batched = []
    all_impression_ids_batched = []

    processed_impressions_count = 0
    total_loss = 0.0

    for batch_features, batch_labels, batch_impression_ids in eval_dataloader:
        # `test_step_fn` updates compiled metrics and returns batch loss + compiled metrics results
        step_results = test_step_fn(model, (batch_features, batch_labels))
        total_loss += step_results["loss"]

        # Get raw predictions from the model for custom metrics calculation
        # The NRMS.call(training=False) should return scores for the slate.
        batch_raw_predictions = model(batch_features, training=False)

        all_labels_batched.append(batch_labels.numpy())
        all_predictions_batched.append(batch_raw_predictions.numpy())
        all_impression_ids_batched.extend(  # Assuming IDs are string tensors or convertable
            [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in batch_impression_ids.numpy()
            ]
        )

        current_batch_size = batch_labels.shape[0]
        processed_impressions_count += current_batch_size
        progress.update(
            eval_progress_task,
            advance=current_batch_size,
            description=f"{mode} (Batch Loss: {step_results['loss']:.4f})",
        )
        if processed_impressions_count >= num_total_impressions:
            break

    progress.remove_task(eval_progress_task)

    if not all_labels_batched:
        console.log(f"[yellow][WARNING {mode}] No data processed in evaluation epoch.[/yellow]")
        # Return default zeroed metrics
        return {m.name: 0.0 for m in model.metrics} | {
            "loss": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "num_impressions": 0,
        }

    # Concatenate results from all batches
    concatenated_labels = np.concatenate(all_labels_batched, axis=0)
    concatenated_predictions = np.concatenate(all_predictions_batched, axis=0)

    # Get final results from compiled Keras metrics
    final_eval_metrics = {m.name: float(m.result().numpy()) for m in model.metrics} | {
        "loss": total_loss / len(all_labels_batched) if all_labels_batched else 0.0
    }

    # Calculate custom metrics (MRR, nDCG) using the custom calculator
    custom_eval_metrics = custom_metrics_calculator.compute_metrics(
        y_true=concatenated_labels, y_pred_logits=concatenated_predictions, progress=progress
    )

    final_eval_metrics = (
        final_eval_metrics | custom_eval_metrics | {"num_impressions": processed_impressions_count}
    )

    if save_predictions_dir:
        predictions_to_save = {
            imp_id: (gt.tolist(), pred.tolist())
            for imp_id, gt, pred in zip(
                all_impression_ids_batched, concatenated_labels, concatenated_predictions
            )
        }
        save_predictions_to_file_fn(predictions_to_save, save_predictions_dir, epoch_idx, mode)

    return final_eval_metrics


def log_metrics_to_console_fn(
    metrics_dict: Dict[str, float], header_prefix: str = ""
) -> None:  # Renamed
    if header_prefix:
        console.log(f"[bold blue]{header_prefix} Metrics:[/bold blue]")
    for name, value in metrics_dict.items():
        try:
            console.log(f"  [green]{name}: {float(value):.4f}[/green]")
        except (ValueError, TypeError):
            console.log(f"  [green]{name}: {value}[/green]")


def save_run_summary_fn(  # Renamed
    summary_output_dir: Path,
    hydra_cfg: DictConfig,
    initial_metrics_dict: Dict[str, float],
    best_metrics_summary_dict: Dict[str, Any],
    test_metrics_dict: Optional[Dict[str, float]] = None,
    wandb_full_history: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Saves training config, key metrics, and history to a JSON file."""
    data_to_save = {
        "configuration": OmegaConf.to_container(hydra_cfg, resolve=True, throw_on_missing=True),
        "initial_validation_metrics": {k: float(v) for k, v in initial_metrics_dict.items()},
        "best_validation_summary": {
            k: (float(v) if isinstance(v, (int, float, np.float32, np.float64)) else v)
            for k, v in best_metrics_summary_dict.items()
        },
    }
    if test_metrics_dict:
        data_to_save["final_test_metrics"] = {k: float(v) for k, v in test_metrics_dict.items()}
    if wandb_full_history:
        data_to_save["wandb_run_history"] = wandb_full_history

    summary_filepath = summary_output_dir / "training_run_summary.json"
    try:
        with open(summary_filepath, "w") as f:
            json.dump(
                data_to_save, f, indent=4, default=lambda o: str(o) if isinstance(o, Path) else None
            )  # Handle Path objects, raise for others
        console.log(f"Training run summary saved to {summary_filepath}")
    except TypeError as e:
        console.log(
            f"[red]Error saving summary to JSON: {e}. Data causing issues might be in complex objects.[/red]"
        )
        # Fallback: try to save parts or print
        console.log(f"Problematic data (partial): {str(data_to_save)[:500]}")


def get_main_comparison_metric(  # Renamed from calculate_validation_metrics
    validation_metrics: Dict[str, float],
    primary_metric_key: str = "ndcg@10",  # Make this configurable from cfg.train.early_stopping_metric
) -> float:
    """Extracts the primary metric used for comparing epochs (e.g., for early stopping)."""
    metric_val = validation_metrics.get(primary_metric_key)
    if metric_val is None:
        # Fallback if primary metric not found, e.g. average of available ones
        console.log(
            f"[yellow]Primary metric '{primary_metric_key}' not found in validation metrics. Using average as fallback.[/yellow]"
        )
        auc = float(validation_metrics.get("auc", 0.0))
        mrr = float(validation_metrics.get("mrr", 0.0))
        ndcg5 = float(validation_metrics.get("ndcg@5", 0.0))
        ndcg10 = float(validation_metrics.get("ndcg@10", 0.0))  # Might be the one missing
        available_metrics = [m for m in [auc, mrr, ndcg5, ndcg10] if m is not None]  # only non-None
        return sum(available_metrics) / len(available_metrics) if available_metrics else 0.0
    return float(metric_val)


def update_best_epoch_metrics(
    best_epoch_summary: Dict[str, Any],
    current_epoch_idx: int,
    current_train_metrics_epoch: Dict[str, float],
    current_val_metrics_epoch: Dict[str, float],
    current_comparison_metric_val: float,
) -> bool:
    """Updates the summary of the best epoch if current epoch is better."""
    is_new_best = False
    if current_comparison_metric_val > best_epoch_summary.get(
        "comparison_metric_value", -float("inf")
    ):
        best_epoch_summary.clear()
        best_epoch_summary |= {
            "epoch_number": current_epoch_idx + 1,
            "train_loss_at_best": float(current_train_metrics_epoch.get("loss", float("nan"))),
            "comparison_metric_value": current_comparison_metric_val,
            **{f"val_{k}": v for k, v in current_val_metrics_epoch.items()},
        }
        is_new_best = True
    return is_new_best


def log_metrics_to_wandb_fn(  # Renamed
    metrics_payload: Dict[str, float],  # Flat dict with prefixes like "train/loss"
    commit_step: int,  # Usually epoch index
    wandb_history_cache: Dict[str, List[float]],  # Local cache for full history
) -> None:
    if wandb.run:  # Check if wandb session is active
        wandb.log(metrics_payload, step=commit_step)
        for key, value in metrics_payload.items():
            wandb_history_cache.setdefault(key, []).append(float(value))


def train_epoch_fn(
    model: tf.keras.Model,
    dataloader_train: tf.data.Dataset,
    num_batches_epoch: int,
    progress_bar: Progress,
    log_interval_steps: int,
    epoch_idx: int,
) -> Dict[str, float]:
    """Trains the model for one epoch."""

    for m in model.metrics:
        m.reset_state()

    sum_batch_losses = 0.0
    epoch_train_task = progress_bar.add_task(
        f"Epoch {epoch_idx + 1}", total=num_batches_epoch, loss="N/A"
    )

    for batch_count, data_batch in enumerate(dataloader_train, 1):
        batch_step_metrics = train_step_fn(model, data_batch)
        current_batch_loss = float(batch_step_metrics["loss"])
        sum_batch_losses += current_batch_loss

        # Only update progress bar and log batch loss
        progress_bar.update(
            epoch_train_task,
            advance=1,
            description=f"Epoch {epoch_idx + 1} (Batch Loss: {current_batch_loss:.4f})",
        )

        if batch_count % log_interval_steps == 0:
            console.log(
                f"Epoch {epoch_idx+1}, Batch {batch_count}/{num_batches_epoch}, "
                f"Batch Loss: {current_batch_loss:.4f}."
            )

    progress_bar.remove_task(epoch_train_task)

    avg_loss = sum_batch_losses / num_batches_epoch if num_batches_epoch > 0 else 0.0

    return {"loss": avg_loss}


def log_epoch_summary_fn(
    current_epoch_idx: int,
    epoch_train_metrics_results: Dict[str, float],
    epoch_val_metrics_results: Dict[str, float],
    wandb_cache: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Logs comprehensive summary for an epoch to console and WandB."""
    console.rule(f"[bold magenta]Epoch {current_epoch_idx + 1} Completed[/bold magenta]")
    log_metrics_to_console_fn(epoch_train_metrics_results, "Average Training")
    log_metrics_to_console_fn(epoch_val_metrics_results, "Validation")
    console.rule()

    if wandb.run and wandb_cache is not None:
        wandb_payload = {f"train/{k}": v for k, v in epoch_train_metrics_results.items()} | {
            f"val/{k}": v for k, v in epoch_val_metrics_results.items()
        }
        log_metrics_to_wandb_fn(wandb_payload, current_epoch_idx + 1, wandb_cache)


def _run_initial_validation(
    model: tf.keras.Model,
    dataset_provider: FastEvalDataProvider,
    custom_metrics_engine: NewsRecommenderMetrics,
    progress_bar_manager: Progress,
    cfg: DictConfig,
) -> Dict[str, float]:
    """Run initial validation before training starts."""
    if not cfg.eval.run_initial_validation:
        return {}

    console.log("[bold yellow]Running Initial Validation (before training starts)...[/bold yellow]")

    # Create mode-specific directory for predictions using Hydra's output directory
    if hydra.core.hydra_config.HydraConfig.initialized():
        output_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        predictions_save_dir = output_run_dir / "predictions" / "initial_val"
        if cfg.eval.save_predictions:
            predictions_save_dir.mkdir(parents=True, exist_ok=True)
    else:
        predictions_save_dir = None

    if cfg.eval.fast_evaluation:
        metrics = model.fast_evaluate(
            behaviors_data=dataset_provider.val_behaviors_data,
            processed_news=dataset_provider.processed_news,
            metrics_calculator=custom_metrics_engine,
            progress=progress_bar_manager,
            batch_size_eval=cfg.eval.batch_size,
            mode="initial_val",
            save_predictions_path=predictions_save_dir if cfg.eval.save_predictions else None,
            epoch=None,
        )
    else:
        metrics = run_evaluation_epoch(
            model,
            dataset_provider.val_dataloader(batch_size=cfg.eval.batch_size),
            custom_metrics_engine,
            dataset_provider.val_size,
            progress_bar_manager,
            mode="initial_val",
            save_predictions_dir=predictions_save_dir if cfg.eval.save_predictions else None,
            epoch_idx=None,
        )
    log_metrics_to_console_fn(metrics, "Initial Validation")
    return metrics


def _run_epoch_evaluation(
    model: tf.keras.Model,
    dataset_provider: FastEvalDataProvider,
    custom_metrics_engine: NewsRecommenderMetrics,
    progress_bar_manager: Progress,
    cfg: DictConfig,
    epoch_idx: int,
    predictions_save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Run evaluation for a single epoch."""
    if cfg.eval.fast_evaluation:
        # Create mode-specific directory for predictions
        mode_specific_dir = predictions_save_dir / "val" if predictions_save_dir else None
        if mode_specific_dir:
            mode_specific_dir.mkdir(parents=True, exist_ok=True)

        return model.fast_evaluate(
            behaviors_data=dataset_provider.val_behaviors_data,
            processed_news=dataset_provider.processed_news,
            metrics_calculator=custom_metrics_engine,
            progress=progress_bar_manager,
            batch_size_eval=cfg.eval.batch_size,
            mode="val",
            save_predictions_path=mode_specific_dir if cfg.eval.save_predictions else None,
            epoch=epoch_idx,
        )

    # Create mode-specific directory for predictions
    mode_specific_dir = predictions_save_dir / "val" if predictions_save_dir else None
    if mode_specific_dir:
        mode_specific_dir.mkdir(parents=True, exist_ok=True)

    return run_evaluation_epoch(
        model,
        dataset_provider.val_dataloader(batch_size=cfg.eval.batch_size),
        custom_metrics_engine,
        dataset_provider.val_size,
        progress_bar_manager,
        mode="val",
        save_predictions_dir=mode_specific_dir if cfg.eval.save_predictions else None,
        epoch_idx=epoch_idx,
    )


def _run_final_testing(
    model: tf.keras.Model,
    dataset_provider: FastEvalDataProvider,
    custom_metrics_engine: NewsRecommenderMetrics,
    progress_bar_manager: Progress,
    cfg: DictConfig,
    best_model_weights_filepath: Path,
    last_model_weights_filepath: Path,
    best_epoch_metrics_tracking: Dict[str, Any],
    predictions_save_dir: Optional[Path] = None,
) -> Optional[Dict[str, float]]:
    """Run final testing phase after training."""
    console.log("Loading best model weights from")
    console.log(f"'{best_model_weights_filepath}' for final testing...")
    model.load_weights(best_model_weights_filepath)
    console.log("Best model weights loaded successfully for testing.")

    # Load test data only when needed
    console.log("Loading test data for final evaluation...")
    dataset_provider._load_data(mode="test")

    # Create mode-specific directory for predictions
    mode_specific_dir = predictions_save_dir / "test" if predictions_save_dir else None
    if mode_specific_dir:
        mode_specific_dir.mkdir(parents=True, exist_ok=True)

    if cfg.eval.fast_evaluation:
        return model.fast_evaluate(
            behaviors_data=dataset_provider.test_behaviors_data,
            processed_news=dataset_provider.processed_news,
            metrics_calculator=custom_metrics_engine,
            progress=progress_bar_manager,
            batch_size_eval=cfg.eval.batch_size,
            mode="test",
            save_predictions_path=mode_specific_dir if cfg.eval.save_predictions else None,
            epoch=best_epoch_metrics_tracking.get("epoch_number", -1) - 1,
        )

    return run_evaluation_epoch(
        model,
        dataset_provider.test_dataloader(batch_size=cfg.eval.batch_size),
        custom_metrics_engine,
        dataset_provider.test_size,
        progress_bar_manager,
        mode="test",
        save_predictions_dir=mode_specific_dir if cfg.eval.save_predictions else None,
        epoch_idx=best_epoch_metrics_tracking.get("epoch_number", -1) - 1,
    )


def _handle_model_saving_and_early_stopping(
    model: tf.keras.Model,
    epoch_idx: int,
    current_metrics: Dict[str, float],
    best_metrics: Dict[str, Any],
    patience_countdown: int,
    cfg: DictConfig,
    best_model_path: Path,
    last_model_path: Path,
) -> Tuple[bool, int]:
    """Handle model saving and early stopping logic.

    Returns:
        Tuple of (should_stop, new_patience_countdown)
    """
    model.save_weights(str(last_model_path))
    current_metric_val = get_main_comparison_metric(
        current_metrics, cfg.train.early_stopping.metric
    )

    if update_best_epoch_metrics(
        best_metrics,
        epoch_idx,
        current_metrics,
        current_metrics,
        current_metric_val,
    ):
        console.log(
            f"[bold green]Validation metric '{cfg.train.early_stopping.metric}' improved to {current_metric_val:.4f}. Saving best model weights.[/bold green]"
        )
        model.save_weights(str(best_model_path))
        return False, cfg.train.early_stopping.patience

    new_patience = patience_countdown - 1
    console.log(
        f"[yellow]Validation metric '{cfg.train.early_stopping.metric}' did not improve from {best_metrics.get('comparison_metric_value', -float('inf')):.4f}. Patience: {new_patience}/{cfg.train.early_stopping.patience}[/yellow]"
    )

    if new_patience <= 0:
        console.log(f"[bold red]Early stopping triggered after epoch {epoch_idx + 1}.[/bold red]")
        return True, new_patience

    return False, new_patience


def training_loop_orchestrator(
    model: tf.keras.Model,
    dataset_provider: FastEvalDataProvider,
    cfg: DictConfig,
    custom_metrics_engine: NewsRecommenderMetrics,
    progress_bar_manager: Progress,
    output_directory: Path,
) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
    """Orchestrates the main training loop, including validation, early stopping, and model saving."""
    num_train_batches = calculate_num_batches(dataset_provider.train_size, cfg.train.batch_size)
    patience_countdown = cfg.train.early_stopping.patience
    best_epoch_metrics_tracking = {"comparison_metric_value": -float("inf")}
    wandb_local_history = {} if cfg.logging.enable_wandb else None

    # Setup directories
    models_save_dir = output_directory / "models"
    models_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_weights_filepath = models_save_dir / "nrms_best.weights.h5"
    last_model_weights_filepath = models_save_dir / "nrms_last.weights.h5"
    predictions_save_dir = output_directory / "predictions"
    predictions_save_dir.mkdir(parents=True, exist_ok=True)

    # Initial validation
    initial_validation_metrics = _run_initial_validation(
        model, dataset_provider, custom_metrics_engine, progress_bar_manager, cfg
    )
    if wandb.run and wandb_local_history is not None:
        log_metrics_to_wandb_fn(
            {f"initial_val/{k}": v for k, v in initial_validation_metrics.items()},
            0,
            wandb_local_history,
        )

    # Main training loop
    overall_training_task = progress_bar_manager.add_task(
        "Overall Training Progress", total=cfg.train.num_epochs
    )

    for epoch_idx in range(cfg.train.num_epochs):
        progress_bar_manager.update(
            overall_training_task,
            description=f"Training Epoch {epoch_idx+1}/{cfg.train.num_epochs}",
        )

        # Training
        final_train_metrics_epoch = train_epoch_fn(
            model,
            dataset_provider.train_dataloader(batch_size=cfg.train.batch_size),
            num_train_batches,
            progress_bar_manager,
            cfg.logging.log_every_n_steps,
            epoch_idx,
        )

        # Validation
        final_val_metrics_epoch = _run_epoch_evaluation(
            model,
            dataset_provider,
            custom_metrics_engine,
            progress_bar_manager,
            cfg,
            epoch_idx,
            predictions_save_dir,
        )

        log_epoch_summary_fn(
            epoch_idx, final_train_metrics_epoch, final_val_metrics_epoch, wandb_local_history
        )

        # Model saving & early stopping
        should_stop, patience_countdown = _handle_model_saving_and_early_stopping(
            model,
            epoch_idx,
            final_val_metrics_epoch,
            best_epoch_metrics_tracking,
            patience_countdown,
            cfg,
            best_model_weights_filepath,
            last_model_weights_filepath,
        )

        if should_stop:
            break

        progress_bar_manager.update(overall_training_task, advance=1)

    progress_bar_manager.remove_task(overall_training_task)

    # Final testing
    final_test_metrics_results = _run_final_testing(
        model,
        dataset_provider,
        custom_metrics_engine,
        progress_bar_manager,
        cfg,
        best_model_weights_filepath,
        last_model_weights_filepath,
        best_epoch_metrics_tracking,
        predictions_save_dir,
    )

    if final_test_metrics_results:
        log_metrics_to_console_fn(final_test_metrics_results, "Final Test")
        if wandb.run and wandb_local_history is not None:
            log_metrics_to_wandb_fn(
                {f"test/{k}": v for k, v in final_test_metrics_results.items()},
                cfg.train.num_epochs + 1,
                wandb_local_history,
            )

    # Save run summary
    save_run_summary_fn(
        output_directory,
        cfg,
        initial_validation_metrics,
        best_epoch_metrics_tracking,
        final_test_metrics_results,
        wandb_local_history,
    )

    if wandb.run:
        wandb.summary["best_epoch"] = best_epoch_metrics_tracking.get("epoch_number")
        wandb.summary["best_val_metric"] = best_epoch_metrics_tracking.get("comparison_metric_value")
        for k, v in best_epoch_metrics_tracking.items():
            wandb.summary[f"best/{k}"] = v
        wandb.finish()
        console.log("Wandb session finished.")

    return best_epoch_metrics_tracking, final_test_metrics_results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main_training_entry(cfg: DictConfig) -> None:
    """Main entry point for training, configured by Hydra."""
    console.log("--- NRMS Training Run Initializing ---")
    console.log("Configuration used:")
    console.log(OmegaConf.to_yaml(cfg))
    console.log("------------------------------------")

    if cfg.device.mixed_precision:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        console.log(f"Mixed precision policy globally set to: {policy.name}")
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Initialize WandB
    setup_wandb_session(cfg)

    # Model and Dataset Initialization
    nrms_model, dataset_provider = initialize_model_and_dataset(cfg)

    # Metrics Calculator
    metrics_engine = NewsRecommenderMetrics(
        **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
    )

    # Setup output directory based on Hydra's current run path
    if hydra.core.hydra_config.HydraConfig.initialized():
        output_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    else:  # Fallback if not using Hydra's launcher
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name_for_path = (
            cfg.logging.experiment_name.replace(" ", "_").lower()
            if cfg.logging.experiment_name
            else "nrms_experiment"
        )
        output_run_dir = Path(cfg.train.output_base_dir) / exp_name_for_path / f"run_{timestamp}"
    output_run_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"All outputs for this run will be saved in: {output_run_dir.resolve()}")

    # Rich Progress Bar context manager
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),  # Auto-width
        TaskProgressColumn(),
        TextColumn("({task.completed} of {task.total} batches)"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as global_progress_bar:

        training_loop_orchestrator(
            nrms_model,
            dataset_provider,
            cfg,
            metrics_engine,
            global_progress_bar,
            output_run_dir,
        )

    console.log("--- NRMS Training Run Finished ---")


if __name__ == "__main__":
    main_training_entry()

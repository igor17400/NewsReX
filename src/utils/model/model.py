import inspect
from typing import Any, Tuple

import hydra
import keras
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from ..training.losses import get_loss

console = Console()


def initialize_model_and_dataset(cfg: DictConfig, training_metrics: list = None) -> Tuple[keras.Model, Any]:
    """Instantiate dataset and model based on Hydra configuration."""
    console.log("Initializing dataset provider...")
    dataset_provider: Any = hydra.utils.instantiate(cfg.dataset, mode="train")
    processed_news_data = dataset_provider.processed_news

    console.log(f"Initializing model: {cfg.model._target_}...")

    # --- Generic Model Instantiation ---
    model_class = hydra.utils.get_class(cfg.model._target_)
    model_signature = inspect.signature(model_class.__init__)

    # Start with parameters from the model's config section
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    # Remove hydra's target, it's not a model parameter
    model_params.pop("_target_", None)

    # Add parameters that are required by the model but are not in its specific config section
    if "processed_news" in model_signature.parameters:
        model_params["processed_news"] = processed_news_data
    if "seed" in model_signature.parameters:
        model_params["seed"] = cfg.seed
    if "max_title_length" in model_signature.parameters:
        model_params["max_title_length"] = cfg.dataset.max_title_length
    if "max_abstract_length" in model_signature.parameters:
        model_params["max_abstract_length"] = cfg.dataset.max_abstract_length
    if "max_history_length" in model_signature.parameters:
        model_params["max_history_length"] = cfg.dataset.max_history_length
    if "max_impressions_length" in model_signature.parameters:
        model_params["max_impressions_length"] = cfg.dataset.max_impressions_length

    # Filter out any parameters that are not in the model's __init__ signature
    valid_params = {k: v for k, v in model_params.items() if k in model_signature.parameters}

    model: keras.Model = model_class(**valid_params)
    # --- End Generic Model Instantiation ---

    console.log(f"Successfully instantiated {model.name} model.")

    optimizer = keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)
    if (keras.mixed_precision.global_policy().name == "mixed_float16"):
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    loss_function = get_loss(
        loss_name=cfg.model.loss.name,
        from_logits=cfg.model.loss.from_logits,
        reduction=cfg.model.loss.reduction,
        label_smoothing=cfg.model.loss.label_smoothing
    )

    # Compile with metrics if provided
    compile_kwargs = {"optimizer": optimizer, "loss": loss_function}
    if training_metrics is not None:
        compile_kwargs["metrics"] = training_metrics

    model.compile(**compile_kwargs)

    metrics_info = f", Metrics: {len(training_metrics)} metrics" if training_metrics else ""
    console.log(
        f"Model compiled. Optimizer: {type(optimizer).__name__}, Loss: {loss_function.name}{metrics_info}"
    )

    console.log(f"Mixed precision global policy: {keras.mixed_precision.global_policy().name}")
    console.log(f"Optimizer type: {type(optimizer)}")

    return model, dataset_provider

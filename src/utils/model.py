import inspect
from typing import Any, Tuple

import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from .losses import get_loss

console = Console()


def initialize_model_and_dataset(cfg: DictConfig) -> Tuple[tf.keras.Model, Any]:
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

    # Filter out any parameters that are not in the model's __init__ signature
    valid_params = {k: v for k, v in model_params.items() if k in model_signature.parameters}

    model: tf.keras.Model = model_class(**valid_params)
    # --- End Generic Model Instantiation ---

    console.log(f"Successfully instantiated {model.name} model.")

    if hasattr(model, "training_model") and model.training_model is not None:
        console.log(f"[bold cyan]Summary of {model.name} Training Model (internal):[/bold cyan]")
        model.training_model.summary(print_fn=lambda s: console.log(s))
    if hasattr(model, "scorer_model") and model.scorer_model is not None:
        console.log(f"[bold cyan]Summary of {model.name} Scorer Model (internal):[/bold cyan]")
        model.scorer_model.summary(print_fn=lambda s: console.log(s))

    console.log(f"[bold cyan]Summary of {model.name} Model (main wrapper):[/bold cyan]")
    model.summary(print_fn=lambda s: console.log(s))

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)
    if (
        cfg.device.mixed_precision
        and tf.keras.mixed_precision.global_policy().name == "mixed_float16"
    ):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    loss_function = get_loss(
        loss_name=cfg.model.loss.name,
        from_logits=cfg.model.loss.from_logits,
        reduction=cfg.model.loss.reduction,
        label_smoothing=cfg.model.loss.label_smoothing
    )

    model.compile(optimizer=optimizer, loss=loss_function)
    console.log(
        f"Model compiled. Optimizer: {type(optimizer).__name__}, Loss: {loss_function.name}"
    )

    console.log(f"Mixed precision global policy: {tf.keras.mixed_precision.global_policy().name}")
    console.log(f"Optimizer type: {type(optimizer)}")

    return model, dataset_provider

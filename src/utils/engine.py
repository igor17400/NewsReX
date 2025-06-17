import tensorflow as tf
from typing import Dict, Tuple

from rich.console import Console

console = Console()


def train_step_fn(
    model: tf.keras.Model, data: Tuple[Dict[str, tf.Tensor], tf.Tensor]
) -> Dict[str, tf.Tensor]:
    """Custom training step logic. Expects data as (features_dict, labels_tensor)."""
    features, labels = data

    with tf.GradientTape() as tape:
        predictions = model(features, training=True)

        # Get the loss function from the model
        loss_fn = model.loss

        # Compute loss using the configured loss function
        loss = loss_fn(y_true=labels, y_pred=predictions, sample_weight=None)

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

    # Get the loss function from the model
    loss_fn = model.loss

    # Compute loss using the configured loss function
    loss = loss_fn(y_true=labels, y_pred=predictions, sample_weight=None)

    # Update metrics
    for metric in model.metrics:
        metric.update_state(labels, predictions)

    batch_metrics = {m.name: m.result() for m in model.metrics}
    batch_metrics["loss"] = loss
    return batch_metrics

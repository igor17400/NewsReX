from typing import Any

import tensorflow as tf


class NewsRecommenderLoss(tf.keras.losses.Loss):
    """Base class for news recommendation losses."""

    def __init__(self, name: str = "news_recommender_loss", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)


class CategoricalCrossEntropyLoss(NewsRecommenderLoss):
    """Categorical cross-entropy loss for training with multiple candidates.

    This loss is used during training when we have multiple candidates and use softmax activation.
    It computes the categorical cross-entropy between the predicted scores and the true labels.

    Args:
        name: Loss name
        from_logits: Whether the predictions are logits or probabilities
    """

    def __init__(
        self, name: str = "categorical_crossentropy", from_logits: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the loss.

        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted scores (probabilities after softmax)

        Returns:
            Loss value
        """
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        )


class BinaryCrossEntropyLoss(NewsRecommenderLoss):
    """Binary cross-entropy loss for validation/testing with single candidate.

    This loss is used during validation/testing when we have a single candidate and use sigmoid activation.
    It computes the binary cross-entropy between the predicted score and the true label.

    Args:
        name: Loss name
        from_logits: Whether the predictions are logits or probabilities
    """

    def __init__(
        self, name: str = "binary_crossentropy", from_logits: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the loss.

        Args:
            y_true: True label (0 or 1)
            y_pred: Predicted score (probability after sigmoid)

        Returns:
            Loss value
        """
        return tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        )


def get_loss(loss_name: str, **kwargs: Any) -> NewsRecommenderLoss:
    """Get a loss function by name.

    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function instance

    Raises:
        ValueError: If the loss name is not recognized
    """
    losses = {
        "categorical_crossentropy": CategoricalCrossEntropyLoss,
        "binary_crossentropy": BinaryCrossEntropyLoss,
    }

    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available losses: {list(losses.keys())}")

    return losses[loss_name](**kwargs)

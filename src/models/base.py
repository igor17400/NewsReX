from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional

import tensorflow as tf
from ..utils.losses import get_loss


class BaseNewsRecommender(tf.keras.Model, ABC):
    """Base class for news recommendation models.

    This class enforces a consistent interface for all news recommendation models:
    1. Training mode: Process multiple candidates with softmax activation
    2. Validation/Testing mode: Process single candidate with sigmoid activation
    3. Consistent input/output handling

    Args:
        name: Model name
        loss: Name of the loss function to use
        loss_kwargs: Additional arguments for the loss function
    """

    def __init__(
        self,
        name: str = "base_news_recommender",
        loss: str = "categorical_crossentropy",
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.model = None  # Training model
        self.scorer = None  # Scoring model
        
        # Initialize loss function
        loss_kwargs = loss_kwargs or {}
        self.loss_fn = get_loss(loss, **loss_kwargs)

    @abstractmethod
    def _build_model(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Build both the training model and the scorer model.

        Returns:
            Tuple containing:
                - Training model: Processes multiple candidates with softmax activation
                - Scoring model: Processes single candidate with sigmoid activation
        """
        pass

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = True) -> tf.Tensor:
        """Forward pass using the appropriate model.

        Args:
            inputs: Dictionary containing model inputs
            training: Whether the model is being called in training mode

        Returns:
            Model predictions
        """
        if training:
            return self.model(inputs)

        # For validation/testing, reshape the candidates to process them all at once
        # Shape of candidate inputs: (batch=1, num_candidates, ...)
        candidate_key = next(k for k in inputs if k.startswith("cand_"))
        _, num_candidates = inputs[candidate_key].shape[:2]

        # Reshape candidates to (num_candidates, 1, ...)
        reshaped_candidates = {
            f"cand_one_{k[5:]}": tf.reshape(v, [num_candidates, 1, -1]) if len(v.shape) == 3
            else tf.reshape(v, [num_candidates, 1])
            for k, v in inputs.items()
            if k.startswith("cand_")
        }

        # Repeat history for each candidate
        repeated_history = {
            k: tf.repeat(v, repeats=num_candidates, axis=0)
            for k, v in inputs.items()
            if not k.startswith("cand_")
        }

        # Process all candidates at once
        adapted_inputs = repeated_history | reshaped_candidates
        scores = self.scorer(adapted_inputs)

        # Reshape scores to (1, num_candidates) to match expected output shape
        return tf.reshape(scores, [1, num_candidates])

    def _update_metrics(self, y: tf.Tensor, y_pred: tf.Tensor, loss: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Update metrics and return their values.

        Args:
            y: True labels
            y_pred: Predicted values
            loss: Loss value

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        for metric in self.metrics:
            metric.update_state(y, y_pred)
            metrics[metric.name] = metric.result()
        metrics['loss'] = loss
        return metrics

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom training step"""
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_fn(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return self._update_metrics(y, y_pred, loss)

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom test step"""
        x, y = data
        y_pred = self(x, training=False)
        loss = self.loss_fn(y, y_pred)
        return self._update_metrics(y, y_pred, loss) 
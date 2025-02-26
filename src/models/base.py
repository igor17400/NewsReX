from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model


class BaseNewsRecommender(Model):
    """Base class for news recommendation models"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom training step"""
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom test step"""
        x, y = data
        y_pred = self(x, training=False)

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


class BaseNewsEncoder(tf.keras.layers.Layer, ABC):
    """Abstract base class for news encoders"""

    def __init__(self, cfg: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg

    @abstractmethod
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        pass


class BaseUserEncoder(tf.keras.layers.Layer, ABC):
    """Abstract base class for user encoders"""

    def __init__(self, cfg: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg

    @abstractmethod
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        pass

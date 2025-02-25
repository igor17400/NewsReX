from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model


class BaseNewsRecommender(Model, ABC):
    """Abstract base class for all news recommendation models."""

    def __init__(self, cfg: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg

    @abstractmethod
    def encode_news(self, news_input: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Encode news articles into vectors."""
        pass

    @abstractmethod
    def encode_user(self, history_input: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Encode user history into user vector."""
        pass

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass of the model."""
        news_input, history_input = inputs

        # Encode candidate news
        news_vector = self.encode_news(news_input, training=training)

        # Encode user from history
        user_vector = self.encode_user(history_input, training=training)

        # Calculate click probability
        click_probability = self.calculate_click_probability(news_vector, user_vector)

        return click_probability

    def calculate_click_probability(
        self, news_vector: tf.Tensor, user_vector: tf.Tensor
    ) -> tf.Tensor:
        """Calculate probability of user clicking the news."""
        logits = tf.reduce_sum(news_vector * user_vector, axis=1)
        return tf.nn.sigmoid(logits)


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

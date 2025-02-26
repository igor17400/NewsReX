from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from models.base import BaseNewsRecommender
from models.layers import AttLayer2, SelfAttention


class NewsEncoder(layers.Layer):
    """News encoder with multi-head self-attention.

    This encoder processes the title embeddings using multi-head self-attention
    and additive attention mechanisms.
    """

    def __init__(
        self,
        multiheads: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout_rate: float = 0.2,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the NewsEncoder.

        Args:
            multiheads (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            attention_hidden_dim (int): Dimension of the hidden layer in the attention mechanism.
            dropout_rate (float): Dropout rate for regularization.
            seed (int): Random seed for reproducibility.
            **kwargs (Any): Additional keyword arguments for the Layer.
        """
        super(NewsEncoder, self).__init__(**kwargs)
        self.multihead_attention = SelfAttention(multiheads, head_dim, seed=seed)
        self.additive_attention = AttLayer2(attention_hidden_dim, seed=seed)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Process title embeddings.

        Args:
            inputs (tf.Tensor): Title embeddings with shape [batch_size, seq_length, embedding_dim].
            training (Optional[bool]): Whether the layer should behave in training mode.

        Returns:
            tf.Tensor: Encoded news vector.
        """
        # Process title
        title_repr = self.dropout(inputs, training=training)
        title_repr = self.multihead_attention([title_repr] * 3)  # Q, K, V are the same
        title_repr = self.dropout(title_repr, training=training)
        news_vector = self.additive_attention(title_repr)

        return news_vector


class UserEncoder(layers.Layer):
    """User encoder with multi-head self-attention.

    This encoder processes the news vectors from the user's history using
    multi-head self-attention and additive attention mechanisms.
    """

    def __init__(
        self,
        multiheads: int,
        head_dim: int,
        attention_hidden_dim: int,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the UserEncoder.

        Args:
            multiheads (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            attention_hidden_dim (int): Dimension of the hidden layer in the attention mechanism.
            seed (int): Random seed for reproducibility.
            **kwargs (Any): Additional keyword arguments for the Layer.
        """
        super(UserEncoder, self).__init__(**kwargs)
        self.multihead_attention = SelfAttention(multiheads, head_dim, seed=seed)
        self.additive_attention = AttLayer2(attention_hidden_dim, seed=seed)

    def call(self, news_vecs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Process news vectors from history.

        Args:
            news_vecs (tf.Tensor): News vectors from history with shape [batch_size, history_length, news_vector_dim].
            training (Optional[bool]): Whether the layer should behave in training mode.

        Returns:
            tf.Tensor: Encoded user vector.
        """
        # Multi-head self attention
        click_repr = self.multihead_attention([news_vecs] * 3)  # Q, K, V are the same

        # Additive attention
        user_vector = self.additive_attention(click_repr)

        return user_vector


class NRMS(BaseNewsRecommender):
    """Neural News Recommendation with Multi-Head Self-Attention"""

    def __init__(
        self,
        multiheads: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout_rate: float = 0.2,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the NRMS model.

        Args:
            multiheads (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            attention_hidden_dim (int): Dimension of the hidden layer in the attention mechanism.
            dropout_rate (float): Dropout rate for regularization.
            seed (int): Random seed for reproducibility.
            **kwargs (Any): Additional keyword arguments for the BaseNewsRecommender.
        """
        super(NRMS, self).__init__(**kwargs)

        # News encoder
        self.news_encoder = NewsEncoder(
            multiheads=multiheads,
            head_dim=head_dim,
            attention_hidden_dim=attention_hidden_dim,
            dropout_rate=dropout_rate,
            seed=seed,
        )

        # User encoder
        self.user_encoder = UserEncoder(
            multiheads=multiheads,
            head_dim=head_dim,
            attention_hidden_dim=attention_hidden_dim,
            seed=seed,
        )

    def build(self, input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> None:
        """Build the model based on input shapes.

        Args:
            input_shape (Tuple[Tuple[int, ...], Tuple[int, ...]]): Shapes of the input tensors for news and history.
        """
        news_shape, history_shape = input_shape
        # Create TimeDistributed wrappers
        self.news_encoder_td = layers.TimeDistributed(self.news_encoder)
        self.history_encoder_td = layers.TimeDistributed(self.news_encoder)
        super().build(input_shape)

    def call(self, inputs: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]], training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass of the NRMS model.

        Args:
            inputs (Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]): Input tensors for impression and history news.
            training (Optional[bool]): Whether the model should behave in training mode.

        Returns:
            tf.Tensor: Click probability for each impression.
        """
        impression_news, history_news = inputs

        # Get title embeddings
        impression_title = impression_news["title"]  # [batch_size, num_impressions, seq_len, emb_dim]
        history_title = history_news["title"]  # [batch_size, history_len, seq_len, emb_dim]

        # Encode impression news
        impression_vector = self.news_encoder_td(impression_title)  # [batch_size, num_impressions, news_vector_dim]

        # Encode history news
        history_vectors = self.history_encoder_td(history_title)  # [batch_size, history_len, news_vector_dim]

        # Encode user from history vectors
        user_vector = self.user_encoder(history_vectors, training=training)  # [batch_size, news_vector_dim]

        # Calculate click probability for each impression
        click_probability = K.sum(
            impression_vector * K.expand_dims(user_vector, axis=1), axis=-1
        )  # [batch_size, num_impressions]
        click_probability = K.sigmoid(click_probability)

        return click_probability

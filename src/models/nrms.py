from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Layer,
    MultiHeadAttention,
    Softmax,
    TimeDistributed,
)

from models.base import BaseNewsRecommender


class NewsEncoder(Layer):
    """News encoder with multi-head self-attention."""

    def __init__(
        self,
        multiheads: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout_rate: float = 0.2,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super(NewsEncoder, self).__init__(**kwargs)
        
        # Multi-head self attention
        self.multihead_attention = MultiHeadAttention(
            num_heads=multiheads,
            key_dim=head_dim,
            seed=seed
        )
        
        # Additive attention
        self.attention_dense = Dense(
            attention_hidden_dim,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
        self.attention_query = Dense(
            1,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
        self.attention_softmax = Softmax(axis=1)  # Softmax over sequence dimension
        
        # Dropout for regularization
        self.dropout = Dropout(dropout_rate, seed=seed)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Process title embeddings."""
        # Apply dropout
        title_repr = self.dropout(inputs, training=training)
        
        # Multi-head self attention
        title_repr = self.multihead_attention(
            query=title_repr,
            key=title_repr,
            value=title_repr
        )
        
        # Apply dropout after attention
        title_repr = self.dropout(title_repr, training=training)
        
        # Additive attention
        attention_hidden = self.attention_dense(title_repr)
        attention_score = self.attention_query(attention_hidden)
        attention_weights = self.attention_softmax(attention_score)
        
        # Weighted sum to get final news vector
        news_vector = tf.reduce_sum(title_repr * attention_weights, axis=1)
        
        return news_vector


class UserEncoder(Layer):
    """User encoder with multi-head self-attention."""

    def __init__(
        self,
        multiheads: int,
        head_dim: int,
        attention_hidden_dim: int,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super(UserEncoder, self).__init__(**kwargs)
        
        # Multi-head self attention
        self.multihead_attention = MultiHeadAttention(
            num_heads=multiheads,
            key_dim=head_dim,
            seed=seed
        )
        
        # Additive attention
        self.attention_dense = Dense(
            attention_hidden_dim,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
        self.attention_query = Dense(
            1,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
        self.attention_softmax = Softmax(axis=1)

    def call(self, news_vecs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Process news vectors from history."""
        # Multi-head self attention
        click_repr = self.multihead_attention(
            query=news_vecs,
            key=news_vecs,
            value=news_vecs
        )
        
        # Additive attention
        attention_hidden = self.attention_dense(click_repr)
        attention_score = self.attention_query(attention_hidden)
        attention_weights = self.attention_softmax(attention_score)
        
        # Weighted sum to get final user vector
        user_vector = tf.reduce_sum(click_repr * attention_weights, axis=1)
        
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
        self.news_encoder_td = TimeDistributed(self.news_encoder)
        self.history_encoder_td = TimeDistributed(self.news_encoder)
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

from typing import Any, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, num_heads: int, head_size: int, dropout: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout = dropout
        self.query_dense = layers.Dense(num_heads * head_size)
        self.key_dense = layers.Dense(num_heads * head_size)
        self.value_dense = layers.Dense(num_heads * head_size)
        self.dropout_layer = layers.Dropout(dropout)

    def call(
        self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None, training: Optional[bool] = None
    ) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]

        # Linear layers
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Reshape to multiple heads
        query = tf.reshape(query, [batch_size, -1, self.num_heads, self.head_size])
        key = tf.reshape(key, [batch_size, -1, self.num_heads, self.head_size])
        value = tf.reshape(value, [batch_size, -1, self.num_heads, self.head_size])

        # Transpose to [batch_size, num_heads, seq_length, head_size]
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])

        # Calculate attention scores
        scale = tf.math.sqrt(tf.cast(self.head_size, tf.float32))
        attention = tf.matmul(query, key, transpose_b=True) / scale

        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)
            attention += (1.0 - tf.cast(mask, tf.float32)) * -1e9

        attention = tf.nn.softmax(attention, axis=-1)
        attention = self.dropout_layer(attention, training=training)

        # Apply attention to values
        outputs = tf.matmul(attention, value)
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        outputs = tf.reshape(outputs, [batch_size, -1, self.num_heads * self.head_size])

        return outputs


class AdditiveSelfAttention(layers.Layer):
    def __init__(self, query_vector_dim: int, dropout: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.query_vector_dim = query_vector_dim
        self.dropout = dropout
        self.query = tf.Variable(
            initial_value=tf.random.normal([1, 1, query_vector_dim]),
            trainable=True,
            name="query_vector",
        )
        self.attention = layers.Dense(query_vector_dim, activation="tanh")
        self.dropout_layer = layers.Dropout(dropout)

    def call(
        self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None, training: Optional[bool] = None
    ) -> tf.Tensor:
        attention = self.attention(inputs)
        attention_weights = tf.matmul(attention, self.query, transpose_b=True)
        attention_weights = tf.squeeze(attention_weights, axis=-1)

        if mask is not None:
            attention_weights += (1.0 - tf.cast(mask, tf.float32)) * -1e9

        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        attention_weights = self.dropout_layer(attention_weights, training=training)

        output = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
        return output


class NewsEncoder(layers.Layer):
    def __init__(
        self,
        word_embedding_dim: int,
        num_attention_heads: int,
        query_vector_dim: int,
        dropout_rate: float,
        max_vocabulary_size: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Word embedding
        self.word_embedding = layers.Embedding(
            input_dim=max_vocabulary_size, output_dim=word_embedding_dim
        )

        # Multi-head self-attention
        self.multihead_attention = MultiHeadSelfAttention(
            num_heads=num_attention_heads,
            head_size=word_embedding_dim // num_attention_heads,
            dropout=dropout_rate,
        )

        # Additive attention
        self.additive_attention = AdditiveSelfAttention(
            query_vector_dim=query_vector_dim, dropout=dropout_rate
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Word embedding
        embedded = self.word_embedding(inputs)

        # Multi-head self-attention
        news_vector = self.multihead_attention(embedded, training=training)

        # Additive attention
        news_vector = self.additive_attention(news_vector, training=training)

        return news_vector


class UserEncoder(layers.Layer):
    def __init__(
        self, num_attention_heads: int, query_vector_dim: int, dropout_rate: float, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.multihead_attention = MultiHeadSelfAttention(
            num_heads=num_attention_heads,
            head_size=query_vector_dim // num_attention_heads,
            dropout=dropout_rate,
        )

        self.additive_attention = AdditiveSelfAttention(
            query_vector_dim=query_vector_dim, dropout=dropout_rate
        )

    def call(self, news_vectors: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Multi-head self-attention
        user_vector = self.multihead_attention(news_vectors, training=training)

        # Additive attention
        user_vector = self.additive_attention(user_vector, training=training)

        return user_vector


class NRMS(Model):
    """Neural News Recommendation with Multi-Head Self-Attention.

    This model uses self-attention mechanisms to model user interests from news
    and behaviors.
    """

    def __init__(
        self,
        word_embedding_dim: int,
        num_attention_heads: int,
        query_vector_dim: int,
        dropout_rate: float,
        max_vocabulary_size: int,
        **kwargs: Any,
    ) -> None:
        """Initialize NRMS model.

        Args:
            word_embedding_dim: Dimension of word embeddings.
            num_attention_heads: Number of attention heads in self-attention.
            query_vector_dim: Dimension of query vector for additive attention.
            dropout_rate: Dropout rate for regularization.
            max_vocabulary_size: Maximum size of vocabulary.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        # Store config
        self.word_embedding_dim = word_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.query_vector_dim = query_vector_dim
        self.dropout_rate = dropout_rate

        # News encoder
        self.news_encoder = NewsEncoder(
            word_embedding_dim=word_embedding_dim,
            num_attention_heads=num_attention_heads,
            query_vector_dim=query_vector_dim,
            dropout_rate=dropout_rate,
            max_vocabulary_size=max_vocabulary_size,
        )

        # User encoder
        self.user_encoder = UserEncoder(
            num_attention_heads=num_attention_heads,
            query_vector_dim=query_vector_dim,
            dropout_rate=dropout_rate,
        )

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None
    ) -> tf.Tensor:
        news_input, history_input = inputs

        # Encode news
        news_vector = self.news_encoder(news_input, training=training)

        # Encode history
        history_vectors = self.news_encoder(history_input, training=training)

        # Encode user
        user_vector = self.user_encoder(history_vectors, training=training)

        # Calculate click probability
        click_probability = tf.reduce_sum(news_vector * user_vector, axis=1)
        click_probability = tf.nn.sigmoid(click_probability)

        return click_probability

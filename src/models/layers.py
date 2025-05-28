import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional, Dict, Any


class AdditiveSelfAttention(layers.Layer):
    """
    Additive Self-Attention layer.

    Computes a context vector as a weighted sum of the input sequence elements.
    The weights are learned using a small feed-forward network.

    Args:
        query_vector_dim (int): Dimension of the query vector in the attention mechanism.
                                This is the output dimension of the first dense layer.
        dropout (float): Dropout rate to apply to the attention mechanism. Defaults to 0.0.
    """

    def __init__(self, query_vector_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.query_vector_dim = query_vector_dim
        self.dropout_rate = dropout

        # Layers will be defined in build() to know input_dim
        self.dense_tanh = None
        self.dense_score = None
        self.dropout_layer = None

    def build(self, input_shape: tf.TensorShape):
        """
        Build the layer's weights.

        Args:
            input_shape: Shape of the input tensor (batch_size, sequence_length, embedding_dim).
        """
        # input_shape[-1] is the embedding_dim or feature_dim of each item in the sequence

        self.dense_tanh = layers.Dense(
            units=self.query_vector_dim, activation="tanh", name="attention_dense_tanh"
        )
        self.dense_score = layers.Dense(
            units=1, activation=None, name="attention_dense_score"  # Raw scores
        )
        if self.dropout_rate > 0.0:
            self.dropout_layer = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(
        self, inputs: tf.Tensor, training: bool = None, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Forward pass of the layer.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, embedding_dim).
            training: Boolean indicating whether the layer should behave in training mode (e.g., for dropout).
            mask: Optional mask tensor of shape (batch_size, sequence_length) with 0s for masked elements.

        Returns:
            Context vector of shape (batch_size, embedding_dim).
        """
        # Process inputs through the first dense layer
        # query_projection shape: (batch_size, sequence_length, query_vector_dim)
        query_projection = self.dense_tanh(inputs)

        if self.dropout_layer is not None:
            query_projection = self.dropout_layer(query_projection, training=training)

        # Compute attention scores
        # attention_scores shape: (batch_size, sequence_length, 1)
        attention_scores = self.dense_score(query_projection)

        if mask is not None:
            # Apply the mask (set scores of masked items to a very small number before softmax)
            mask = tf.expand_dims(
                tf.cast(mask, tf.float32), axis=-1
            )  # (batch_size, sequence_length, 1)
            attention_scores += mask * -1e9  # Add a large negative number to masked positions

        # Compute attention weights using softmax
        # attention_weights shape: (batch_size, sequence_length, 1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)

        # Compute the context vector as a weighted sum of the input sequence
        # context_vector shape: (batch_size, embedding_dim)
        return tf.reduce_sum(attention_weights * inputs, axis=1)

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the layer."""
        config = super().get_config()
        config.update(
            {
                "query_vector_dim": self.query_vector_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config

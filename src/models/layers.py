# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
#
# This code has been updated to use the modern Keras 3 API.
# The SelfAttention layer has been refactored to use the built-in
# keras.layers.MultiHeadAttention for better performance and maintainability.

import keras
from keras import layers
from keras import ops


class AdditiveAttentionLayer(layers.Layer):
    """
    Soft-alignment-based attention layer.

    This layer computes a weighted sum of the input sequence, where the weights are
    learned during training. It's a common attention mechanism used in various
    NLP and recommendation models.

    Args:
        dim (int): The hidden dimension of the attention mechanism.
        seed (int): Random seed for reproducibility of initializers.
    """

    def __init__(self, query_vec_dim=200, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = query_vec_dim
        self.seed = seed
        self.supports_masking = True

    def build(self, input_shape):
        """
        Create the weights for the layer.

        This method creates the three trainable variables: W, b, and q, which are
        used to compute the attention scores.
        """
        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError("A `AttLayer2` layer should be called on a 3D tensor.")

        # Weight matrix for the dense transformation
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], self.dim),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        # Bias vector for the dense transformation
        self.b = self.add_weight(
            name="b",
            shape=(self.dim,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        # Query vector to compute attention scores
        self.q = self.add_weight(
            name="q",
            shape=(self.dim, 1),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        """
        Defines the forward pass of the layer.

        Args:
            inputs (keras.KerasTensor): The input 3D tensor `(batch_size, time_steps, features)`.
            mask (keras.KerasTensor, optional): A boolean mask `(batch_size, time_steps)`
                                                to ignore certain time steps.

        Returns:
            keras.KerasTensor: A 2D tensor `(batch_size, features)` representing the
                weighted sum of the input sequence.
        """
        # 1. Dense transformation and non-linearity
        attention_hidden = ops.tanh(ops.matmul(inputs, self.W) + self.b)

        # 2. Compute attention scores
        attention_scores = ops.matmul(attention_hidden, self.q)
        attention_scores = ops.squeeze(attention_scores, axis=-1)

        # 3. Apply exp and mask (following original paper implementation)
        if mask is None:
            attention = ops.exp(attention_scores)
        else:
            attention = ops.exp(attention_scores) * ops.cast(mask, dtype=self.compute_dtype)

        # 4. Normalize attention weights
        attention_weights = attention / (
                ops.sum(attention, axis=-1, keepdims=True) + keras.backend.epsilon()
        )

        # 5. Compute weighted sum of inputs
        attention_weights_expanded = ops.expand_dims(attention_weights, axis=-1)
        weighted_input = inputs * attention_weights_expanded

        return ops.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        return input_shape[0], input_shape[-1]


class ComputeMasking(layers.Layer):
    """Compute if inputs contains zero value.

    Returns:
        bool tensor: True for values not equal to zero.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """Call method for ComputeMasking.

        Args:
            inputs (object): input tensor.

        Returns:
            bool tensor: True for values not equal to zero.
        """
        mask = ops.not_equal(inputs, 0)
        return ops.cast(mask, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


class OverwriteMasking(layers.Layer):
    """Set values at specific positions to zero.

    Args:
        inputs (list): value tensor and mask tensor.

    Returns:
        object: tensor after setting values to zero.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """Call method for OverwriteMasking.

        Args:
            inputs (list): value tensor and mask tensor.

        Returns:
            object: tensor after setting values to zero.
        """
        return inputs[0] * ops.expand_dims(inputs[1], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

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
        import numpy as np
        
        # DEBUG: Check inputs to AdditiveAttentionLayer
        # TODO: Remove debug prints after verifying the NaN issue is fixed
        inputs_np = ops.convert_to_numpy(inputs)
        if np.isnan(inputs_np).any() or np.isinf(inputs_np).any():
            print(f"DEBUG AdditiveAttention: inputs has NaN/Inf: {np.isnan(inputs_np).any()}/{np.isinf(inputs_np).any()}")
            print(f"DEBUG AdditiveAttention: inputs min/max: {np.min(inputs_np):.6f} / {np.max(inputs_np):.6f}")
        
        # DEBUG: Check weights
        W_np = ops.convert_to_numpy(self.W)
        b_np = ops.convert_to_numpy(self.b)
        q_np = ops.convert_to_numpy(self.q)
        
        if np.isnan(W_np).any() or np.isinf(W_np).any():
            print(f"DEBUG AdditiveAttention: W has NaN/Inf: {np.isnan(W_np).any()}/{np.isinf(W_np).any()}")
            print(f"DEBUG AdditiveAttention: W min/max: {np.min(W_np):.6f} / {np.max(W_np):.6f}")
        
        if np.isnan(b_np).any() or np.isinf(b_np).any():
            print(f"DEBUG AdditiveAttention: b has NaN/Inf: {np.isnan(b_np).any()}/{np.isinf(b_np).any()}")
            print(f"DEBUG AdditiveAttention: b min/max: {np.min(b_np):.6f} / {np.max(b_np):.6f}")
            
        if np.isnan(q_np).any() or np.isinf(q_np).any():
            print(f"DEBUG AdditiveAttention: q has NaN/Inf: {np.isnan(q_np).any()}/{np.isinf(q_np).any()}")
            print(f"DEBUG AdditiveAttention: q min/max: {np.min(q_np):.6f} / {np.max(q_np):.6f}")
        
        # 1. Dense transformation and non-linearity
        attention_hidden = ops.tanh(ops.matmul(inputs, self.W) + self.b)

        # 2. Compute attention scores
        attention_scores = ops.matmul(attention_hidden, self.q)
        attention_scores = ops.squeeze(attention_scores, axis=-1)
        
        # DEBUG: Check attention scores
        scores_np = ops.convert_to_numpy(attention_scores)
        if np.isnan(scores_np).any() or np.isinf(scores_np).any():
            print(f"DEBUG AdditiveAttention: attention_scores has NaN/Inf: {np.isnan(scores_np).any()}/{np.isinf(scores_np).any()}")
            print(f"DEBUG AdditiveAttention: attention_scores min/max: {np.min(scores_np):.6f} / {np.max(scores_np):.6f}")

        # 3. Apply exp and mask (following original paper implementation)
        if mask is None:
            attention = ops.exp(attention_scores)
        else:
            attention = ops.exp(attention_scores) * ops.cast(mask, dtype=self.compute_dtype)
            
        # DEBUG: Check attention after exp
        attention_np = ops.convert_to_numpy(attention)
        if np.isnan(attention_np).any() or np.isinf(attention_np).any():
            print(f"DEBUG AdditiveAttention: attention has NaN/Inf: {np.isnan(attention_np).any()}/{np.isinf(attention_np).any()}")
            print(f"DEBUG AdditiveAttention: attention min/max: {np.min(attention_np):.6f} / {np.max(attention_np):.6f}")

        # 4. Normalize attention weights
        attention_sum = ops.sum(attention, axis=-1, keepdims=True)
        
        # DEBUG: Check attention sum
        sum_np = ops.convert_to_numpy(attention_sum)
        if np.isnan(sum_np).any() or np.isinf(sum_np).any() or (sum_np == 0).any():
            print(f"DEBUG AdditiveAttention: attention_sum has NaN/Inf/Zero: {np.isnan(sum_np).any()}/{np.isinf(sum_np).any()}/{(sum_np == 0).any()}")
            print(f"DEBUG AdditiveAttention: attention_sum min/max: {np.min(sum_np):.6f} / {np.max(sum_np):.6f}")
        
        # Handle the case when all inputs are masked (attention_sum = 0)
        # In this case, return uniform attention weights instead of NaN
        epsilon = keras.backend.epsilon()
        attention_weights = ops.where(
            attention_sum > epsilon,
            attention / (attention_sum + epsilon),
            ops.ones_like(attention) / ops.cast(ops.shape(attention)[-1], attention.dtype)
        )
        
        # DEBUG: Check final attention weights
        weights_np = ops.convert_to_numpy(attention_weights)
        if np.isnan(weights_np).any() or np.isinf(weights_np).any():
            print(f"DEBUG AdditiveAttention: attention_weights has NaN/Inf: {np.isnan(weights_np).any()}/{np.isinf(weights_np).any()}")
            print(f"DEBUG AdditiveAttention: attention_weights min/max: {np.min(weights_np):.6f} / {np.max(weights_np):.6f}")

        # 5. Compute weighted sum of inputs
        attention_weights_expanded = ops.expand_dims(attention_weights, axis=-1)
        weighted_input = inputs * attention_weights_expanded

        result = ops.sum(weighted_input, axis=1)
        
        # DEBUG: Check final result
        result_np = ops.convert_to_numpy(result)
        if np.isnan(result_np).any() or np.isinf(result_np).any():
            print(f"DEBUG AdditiveAttention: result has NaN/Inf: {np.isnan(result_np).any()}/{np.isinf(result_np).any()}")
            print(f"DEBUG AdditiveAttention: result min/max: {np.min(result_np):.6f} / {np.max(result_np):.6f}")

        return result

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

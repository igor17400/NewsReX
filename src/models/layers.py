# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
# Reference: https://github.com/recommenders-team/recommenders/blob/main/recommenders/models/newsrec/models/layers.py

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers


class AttLayer2(layers.Layer):
    """Soft alignment attention implementation.

    Attributes:
        dim (int): Attention hidden dimension.
    """

    def __init__(self, dim: int = 200, seed: int = 0, **kwargs: Any) -> None:
        """Initialize AttLayer2.

        Args:
            dim (int): Attention hidden dimension.
            seed (int): Random seed for weight initialization.
        """
        self.dim = dim
        self.seed = seed
        super(AttLayer2, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Initialize variables in AttLayer2.

        There are three variables in AttLayer2: W, b, and q.

        Args:
            input_shape (tf.TensorShape): Shape of input tensor.
        """
        assert len(input_shape) == 3
        dim = self.dim
        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[-1]), dim),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(dim,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.q = self.add_weight(
            name="q",
            shape=(dim, 1),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        super(AttLayer2, self).build(input_shape)

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None, **kwargs: Any) -> tf.Tensor:
        """Core implementation of soft attention.

        Args:
            inputs (tf.Tensor): Input tensor.
            mask (Optional[tf.Tensor]): Optional mask tensor.

        Returns:
            tf.Tensor: Weighted sum of input tensors.
        """
        attention = K.tanh(K.dot(inputs, self.W) + self.b)
        attention = K.dot(attention, self.q)

        attention = tf.squeeze(attention, axis=2)

        if mask is None:
            attention = tf.exp(attention)
        else:
            attention = tf.exp(attention) * tf.cast(mask, dtype="float32")

        attention_weight = attention / (tf.reduce_sum(attention, axis=-1, keepdims=True) + K.epsilon())

        attention_weight = tf.expand_dims(attention_weight, axis=-1)
        weighted_input = inputs * attention_weight
        return tf.reduce_sum(weighted_input, axis=1)

    def compute_mask(self, input: tf.Tensor, input_mask: Optional[tf.Tensor] = None) -> None:
        """Compute output mask value.

        Args:
            input (tf.Tensor): Input tensor.
            input_mask (Optional[tf.Tensor]): Input mask.

        Returns:
            None: Output mask.
        """
        return None

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """Compute shape of output tensor.

        Args:
            input_shape (Tuple[int, ...]): Shape of input tensor.

        Returns:
            Tuple[int, int]: Shape of output tensor.
        """
        return input_shape[0], input_shape[-1]


class SelfAttention(layers.Layer):
    """Multi-head self-attention implementation.

    Args:
        multiheads (int): The number of heads.
        head_dim (int): Dimension of each head.
        mask_right (bool): Whether to mask right words.

    Returns:
        tf.Tensor: Weighted sum after attention.
    """

    def __init__(self, multiheads: int, head_dim: int, seed: int = 0, mask_right: bool = False, **kwargs: Any) -> None:
        """Initialize SelfAttention.

        Args:
            multiheads (int): The number of heads.
            head_dim (int): Dimension of each head.
            mask_right (bool): Whether to mask right words.
            seed (int): Random seed for weight initialization.
        """
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        super(SelfAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tuple[int, int, int]:
        """Compute shape of output tensor.

        Args:
            input_shape (Tuple[Tuple[int, ...], Tuple[int, ...]]): Shape of input tensor.

        Returns:
            Tuple[int, int, int]: Output shape tuple.
        """
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def build(self, input_shape: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]) -> None:
        """Initialize variables in SelfAttention.

        There are three variables in SelfAttention: WQ, WK, and WV.
        WQ is used for linear transformation of query.
        WK is used for linear transformation of key.
        WV is used for linear transformation of value.

        Args:
            input_shape (Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]): Shape of input tensor.
        """
        self.WQ = self.add_weight(
            name="WQ",
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.WK = self.add_weight(
            name="WK",
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.WV = self.add_weight(
            name="WV",
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        super(SelfAttention, self).build(input_shape)

    def Mask(self, inputs: tf.Tensor, seq_len: Optional[tf.Tensor], mode: str = "add") -> tf.Tensor:
        """Mask operation used in multi-head self-attention.

        Args:
            inputs (tf.Tensor): Input tensor.
            seq_len (Optional[tf.Tensor]): Sequence length of inputs.
            mode (str): Mode of mask.

        Returns:
            tf.Tensor: Tensors after masking.
        """
        if seq_len is None:
            return inputs
        else:
            mask = tf.one_hot(indices=seq_len[:, 0], num_classes=tf.shape(inputs)[1])
            mask = 1 - tf.cumsum(mask, axis=1)

            for _ in range(len(inputs.shape) - 2):
                mask = tf.expand_dims(mask, 2)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]) -> tf.Tensor:
        """Core logic of multi-head self-attention.

        Args:
            QKVs (Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]):
            Inputs of multi-head self-attention i.e. query, key, value, and optionally sequence lengths.

        Returns:
            tf.Tensor: Output tensors.
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs  # type: ignore
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        else:
            raise ValueError("QKVs must be a tuple of length 3 or 5.")

        Q_seq = tf.matmul(Q_seq, self.WQ)
        Q_seq = tf.reshape(Q_seq, shape=(-1, tf.shape(Q_seq)[1], self.multiheads, self.head_dim))
        Q_seq = tf.transpose(Q_seq, perm=(0, 2, 1, 3))

        K_seq = tf.matmul(K_seq, self.WK)
        K_seq = tf.reshape(K_seq, shape=(-1, tf.shape(K_seq)[1], self.multiheads, self.head_dim))
        K_seq = tf.transpose(K_seq, perm=(0, 2, 1, 3))

        V_seq = tf.matmul(V_seq, self.WV)
        V_seq = tf.reshape(V_seq, shape=(-1, tf.shape(V_seq)[1], self.multiheads, self.head_dim))
        V_seq = tf.transpose(V_seq, perm=(0, 2, 1, 3))

        A = tf.einsum("abij, abkj -> abik", Q_seq, K_seq) / tf.sqrt(tf.cast(self.head_dim, dtype="float32"))
        A = tf.transpose(A, perm=(0, 3, 2, 1))

        A = self.Mask(A, V_len, "add")
        A = tf.transpose(A, perm=(0, 3, 2, 1))

        if self.mask_right:
            ones = tf.ones_like(A[:1, :1])
            lower_triangular = tf.linalg.band_part(ones, -1, 0)
            mask = (ones - lower_triangular) * 1e12
            A = A - mask
        A = tf.nn.softmax(A)

        O_seq = tf.einsum("abij, abjk -> abik", A, V_seq)
        O_seq = tf.transpose(O_seq, perm=(0, 2, 1, 3))

        O_seq = tf.reshape(O_seq, shape=(-1, tf.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, "mul")
        return O_seq

    def get_config(self) -> Dict[str, Any]:
        """Add multiheads, head_dim, and mask_right into layer config.

        Returns:
            Dict[str, Any]: Config of SelfAttention layer.
        """
        config = super(SelfAttention, self).get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
            }
        )
        return config


def PersonalizedAttentivePooling(dim1: int, dim2: int, dim3: int, seed: int = 0) -> tf.keras.Model:
    """Soft alignment attention implementation.

    Args:
        dim1 (int): First dimension of value shape.
        dim2 (int): Second dimension of value shape.
        dim3 (int): Shape of query.
        seed (int): Random seed for weight initialization.

    Returns:
        tf.keras.Model: Weighted summary of inputs value.
    """
    vecs_input = tf.keras.Input(shape=(dim1, dim2), dtype="float32")
    query_input = tf.keras.Input(shape=(dim3,), dtype="float32")

    user_vecs = layers.Dropout(0.2)(vecs_input)
    user_att = layers.Dense(
        dim3,
        activation="tanh",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros(),
    )(user_vecs)
    user_att2 = layers.Dot(axes=-1)([query_input, user_att])
    user_att2 = layers.Activation("softmax")(user_att2)
    user_vec = layers.Dot((1, 1))([user_vecs, user_att2])

    model = tf.keras.Model([vecs_input, query_input], user_vec)
    return model


class ComputeMasking(layers.Layer):
    """Compute if inputs contain zero value.

    Returns:
        tf.Tensor: Boolean tensor, True for values not equal to zero.
    """

    def __init__(self, **kwargs: Any) -> None:
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Call method for ComputeMasking.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Boolean tensor, True for values not equal to zero.
        """
        mask = tf.not_equal(inputs, 0)
        return tf.cast(mask, tf.floatx())

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute output shape.

        Args:
            input_shape (Tuple[int, ...]): Shape of input tensor.

        Returns:
            Tuple[int, ...]: Shape of output tensor.
        """
        return input_shape


class OverwriteMasking(layers.Layer):
    """Set values at specific positions to zero.

    Args:
        inputs (Tuple[tf.Tensor, tf.Tensor]): Value tensor and mask tensor.

    Returns:
        tf.Tensor: Tensor after setting values to zero.
    """

    def __init__(self, **kwargs: Any) -> None:
        super(OverwriteMasking, self).__init__(**kwargs)

    def build(self, input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> None:
        """Build method for OverwriteMasking.

        Args:
            input_shape (Tuple[Tuple[int, ...], Tuple[int, ...]]): Shape of input tensors.
        """
        super(OverwriteMasking, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs: Any) -> tf.Tensor:
        """Call method for OverwriteMasking.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Value tensor and mask tensor.

        Returns:
            tf.Tensor: Tensor after setting values to zero.
        """
        return inputs[0] * tf.expand_dims(inputs[1])

    def compute_output_shape(self, input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tuple[int, ...]:
        """Compute output shape.

        Args:
            input_shape (Tuple[Tuple[int, ...], Tuple[int, ...]]): Shape of input tensors.

        Returns:
            Tuple[int, ...]: Shape of output tensor.
        """
        return input_shape[0]

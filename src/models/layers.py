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


class GraphSAGELayer(layers.Layer):
    """GraphSAGE layer implementation for Keras 3.
    
    This layer implements the GraphSAGE algorithm for graph neural networks,
    adapted for use with Keras 3 and JAX backend.
    
    Args:
        units: Output dimension
        num_layers: Number of GraphSAGE layers to stack
        aggregator: Type of aggregation ('mean', 'max', 'sum')
        dropout_rate: Dropout rate
        activation: Activation function
        seed: Random seed
    """

    def __init__(
            self,
            units,
            num_layers=1,
            aggregator='mean',
            dropout_rate=0.0,
            activation='relu',
            seed=42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.aggregator = aggregator
        self.dropout_rate = dropout_rate
        self.activation = keras.activations.get(activation)
        self.seed = seed

    def build(self, input_shape):
        """Build the layer weights."""
        # Input shape: (node_features_shape, edge_index_shape)
        node_features_shape = input_shape[0]
        input_dim = node_features_shape[-1]

        # Create weights for each GraphSAGE layer
        self.W_self = []
        self.W_neigh = []
        self.bias = []

        for i in range(self.num_layers):
            # Weight matrix for self features
            self.W_self.append(
                self.add_weight(
                    name=f'W_self_{i}',
                    shape=(input_dim if i == 0 else self.units, self.units),
                    initializer=keras.initializers.GlorotUniform(seed=self.seed),
                    trainable=True
                )
            )

            # Weight matrix for neighbor features
            self.W_neigh.append(
                self.add_weight(
                    name=f'W_neigh_{i}',
                    shape=(input_dim if i == 0 else self.units, self.units),
                    initializer=keras.initializers.GlorotUniform(seed=self.seed),
                    trainable=True
                )
            )

            # Bias
            self.bias.append(
                self.add_weight(
                    name=f'bias_{i}',
                    shape=(self.units,),
                    initializer=keras.initializers.Zeros(),
                    trainable=True
                )
            )

        self.dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        super().build(input_shape)

    def aggregate_neighbors(self, node_features, adjacency_matrix):
        """Aggregate neighbor features based on the specified method."""
        # adjacency_matrix shape: (batch_size, num_nodes, num_nodes)
        # node_features shape: (batch_size, num_nodes, feature_dim)

        if self.aggregator == 'mean':
            # Mean aggregation
            neighbor_sum = ops.matmul(adjacency_matrix, node_features)
            degree = ops.sum(adjacency_matrix, axis=-1, keepdims=True)
            degree = ops.maximum(degree, 1.0)  # Avoid division by zero
            neighbor_features = neighbor_sum / degree
        elif self.aggregator == 'max':
            # Max aggregation - expand and use maximum
            expanded_features = ops.expand_dims(node_features, axis=1)
            expanded_adj = ops.expand_dims(adjacency_matrix, axis=-1)
            masked_features = expanded_features * expanded_adj
            neighbor_features = ops.max(masked_features, axis=2)
        elif self.aggregator == 'sum':
            # Sum aggregation
            neighbor_features = ops.matmul(adjacency_matrix, node_features)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        return neighbor_features

    def call(self, inputs, training=None):
        """Forward pass of GraphSAGE layer.
        
        Args:
            inputs: Tuple of (node_features, adjacency_matrix)
                - node_features: (batch_size, num_nodes, feature_dim)
                - adjacency_matrix: (batch_size, num_nodes, num_nodes)
            training: Whether in training mode
            
        Returns:
            Updated node features: (batch_size, num_nodes, units)
        """
        node_features, adjacency_matrix = inputs
        h = node_features

        for i in range(self.num_layers):
            # Aggregate neighbor features
            h_neigh = self.aggregate_neighbors(h, adjacency_matrix)

            # Transform self and neighbor features
            h_self = ops.matmul(h, self.W_self[i])
            h_neigh = ops.matmul(h_neigh, self.W_neigh[i])

            # Combine self and neighbor features
            h = h_self + h_neigh + self.bias[i]

            # Apply activation and dropout for all layers
            # Remove conditional logic to avoid JAX tracing issues
            h = self.activation(h)
            h = self.dropout(h, training=training)

        return h

    def compute_output_shape(self, input_shape):
        node_features_shape = input_shape[0]
        return (node_features_shape[0], node_features_shape[1], self.units)


class MultiHeadAttentionBlock(layers.Layer):
    """Multi-head attention block for set-based operations.
    
    This implements the MAB (Multihead Attention Block) used in Set Transformer.
    
    Args:
        dim_out: Output dimension
        num_heads: Number of attention heads
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(self, dim_out, num_heads, use_layer_norm=True, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.use_layer_norm = use_layer_norm
        self.seed = seed

    def build(self, input_shape):
        # Expecting input_shape to be a tuple of (Q_shape, K_shape)
        if isinstance(input_shape, tuple) and len(input_shape) == 2:
            q_shape, k_shape = input_shape
        else:
            # Self-attention case
            q_shape = k_shape = input_shape

        self.fc_q = layers.Dense(self.dim_out, use_bias=True,
                                 kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed))
        self.fc_k = layers.Dense(self.dim_out, use_bias=True,
                                 kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed))
        self.fc_v = layers.Dense(self.dim_out, use_bias=True,
                                 kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed))
        self.fc_o = layers.Dense(self.dim_out, use_bias=True,
                                 kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed))

        if self.use_layer_norm:
            self.ln0 = layers.LayerNormalization()
            self.ln1 = layers.LayerNormalization()

        self.dropout = layers.Dropout(0.1)
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: Either a single tensor for self-attention or tuple (Q, K) for cross-attention
            training: Whether in training mode
            
        Returns:
            Attention output with same shape as Q
        """
        if isinstance(inputs, tuple):
            Q, K = inputs
        else:
            Q = K = inputs

        # Linear projections
        Q_proj = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V_proj = self.fc_v(K)

        # Get shapes for multi-head attention
        # Using static shapes where possible for JAX compatibility
        q_shape = ops.shape(Q)
        k_shape = ops.shape(K)
        batch_size = q_shape[0]
        seq_len_q = q_shape[1]  
        seq_len_k = k_shape[1]

        head_dim = self.dim_out // self.num_heads

        Q_proj = ops.reshape(Q_proj, (batch_size, seq_len_q, self.num_heads, head_dim))
        K_proj = ops.reshape(K_proj, (batch_size, seq_len_k, self.num_heads, head_dim))
        V_proj = ops.reshape(V_proj, (batch_size, seq_len_k, self.num_heads, head_dim))

        # Transpose for attention computation
        Q_proj = ops.transpose(Q_proj, (0, 2, 1, 3))  # (batch, heads, seq_q, head_dim)
        K_proj = ops.transpose(K_proj, (0, 2, 1, 3))  # (batch, heads, seq_k, head_dim)
        V_proj = ops.transpose(V_proj, (0, 2, 1, 3))  # (batch, heads, seq_k, head_dim)

        # Compute attention scores
        scores = ops.matmul(Q_proj, ops.transpose(K_proj, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(ops.cast(head_dim, scores.dtype))

        # Apply softmax
        attention_weights = ops.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention to values
        attention_output = ops.matmul(attention_weights, V_proj)

        # Reshape back
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = ops.reshape(attention_output, (batch_size, seq_len_q, self.dim_out))

        # Output projection and residual connection
        O = self.fc_o(attention_output)
        O = self.dropout(O, training=training)

        # Add residual connection
        O = O + Q

        # Layer normalization
        if self.use_layer_norm:
            O = self.ln0(O)

        # Feed-forward and another residual
        O_ff = self.fc_o(ops.relu(O))
        O = O + O_ff

        if self.use_layer_norm:
            O = self.ln1(O)

        return O

    def compute_output_shape(self, input_shape):
        return input_shape


class InducedSetAttentionBlock(layers.Layer):
    """Induced Set Attention Block (ISAB) for set-based operations.
    
    This implements the ISAB layer from the Set Transformer paper, which uses
    inducing points to reduce computational complexity.
    
    Args:
        dim_out: Output dimension
        num_heads: Number of attention heads
        num_inducing_points: Number of inducing points
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(self, dim_out, num_heads, num_inducing_points, use_layer_norm=True, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.num_inducing_points = num_inducing_points
        self.use_layer_norm = use_layer_norm
        self.seed = seed

    def build(self, input_shape):
        # Initialize inducing points
        self.inducing_points = self.add_weight(
            name='inducing_points',
            shape=(1, self.num_inducing_points, self.dim_out),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True
        )

        # Create two MAB blocks
        self.mab0 = MultiHeadAttentionBlock(self.dim_out, self.num_heads, self.use_layer_norm, self.seed)
        self.mab1 = MultiHeadAttentionBlock(self.dim_out, self.num_heads, self.use_layer_norm, self.seed)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass of ISAB.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, dim)
            training: Whether in training mode
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim_out)
        """
        # Use broadcasting instead of repeat to avoid dynamic batch size issues
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        
        # Broadcast inducing points for batch
        I = ops.broadcast_to(
            ops.expand_dims(self.inducing_points, axis=0),
            (batch_size, self.inducing_points.shape[0], self.inducing_points.shape[1])
        )

        # First MAB: attending from inducing points to input
        H = self.mab0((I, inputs), training=training)

        # Second MAB: attending from input to inducing points
        output = self.mab1((inputs, H), training=training)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.dim_out,)

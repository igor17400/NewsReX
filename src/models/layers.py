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
    """GraphSAGE layer implementation for CROWN paper.
    
    This layer implements GraphSAGE specifically for the bipartite user-news graph
    used in the CROWN model, with mutual updates between user and news embeddings.
    
    Args:
        units: Output dimension for both user and news embeddings
        aggregator: Type of aggregation ('mean', 'max', 'sum', 'attention')
        dropout_rate: Dropout rate
        activation: Activation function
        seed: Random seed
        normalize: Whether to L2-normalize the output embeddings
    """

    def __init__(
            self,
            units,
            aggregator='mean',
            dropout_rate=0.0,
            activation='relu',
            seed=42,
            normalize=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.aggregator = aggregator
        self.dropout_rate = dropout_rate
        self.activation = keras.activations.get(activation)
        self.seed = seed
        self.normalize = normalize

    def build(self, input_shape):
        """Build the layer weights.
        
        Expected input shapes:
        - user_features: (batch_size, num_users, user_feature_dim)
        - news_features: (batch_size, num_news, news_feature_dim)
        - adjacency_matrix: (batch_size, num_users + num_news, num_users + num_news)
        """
        if len(input_shape) != 3:
            raise ValueError("GraphSAGELayer expects 3 inputs: (user_features, news_features, adjacency_matrix)")

        user_features_shape, news_features_shape, _ = input_shape
        user_dim = user_features_shape[-1]
        news_dim = news_features_shape[-1]

        # User update weights: W_user_self, W_user_neigh (from news)
        self.W_user_self = self.add_weight(
            name='W_user_self',
            shape=(user_dim, self.units),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True
        )

        self.W_user_neigh = self.add_weight(
            name='W_user_neigh',
            shape=(news_dim, self.units),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True
        )

        self.b_user = self.add_weight(
            name='b_user',
            shape=(self.units,),
            initializer=keras.initializers.Zeros(),
            trainable=True
        )

        # News update weights: W_news_self, W_news_neigh (from users)
        self.W_news_self = self.add_weight(
            name='W_news_self',
            shape=(news_dim, self.units),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True
        )

        self.W_news_neigh = self.add_weight(
            name='W_news_neigh',
            shape=(user_dim, self.units),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True
        )

        self.b_news = self.add_weight(
            name='b_news',
            shape=(self.units,),
            initializer=keras.initializers.Zeros(),
            trainable=True
        )

        # Attention weights if using attention aggregator
        if self.aggregator == 'attention':
            self.W_att = self.add_weight(
                name='W_att',
                shape=(self.units * 2, 1),
                initializer=keras.initializers.GlorotUniform(seed=self.seed),
                trainable=True
            )

        self.dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        super().build(input_shape)

    def aggregate_neighbors(self, node_features, neighbor_features, adjacency_weights):
        """Aggregate neighbor features using specified aggregator.
        
        Args:
            node_features: Features of target nodes (batch_size, num_nodes, feature_dim)
            neighbor_features: Features of neighbor nodes (batch_size, num_neighbors, feature_dim)  
            adjacency_weights: Adjacency weights (batch_size, num_nodes, num_neighbors)
        
        Returns:
            Aggregated neighbor features (batch_size, num_nodes, feature_dim)
        """
        if self.aggregator == 'mean':
            # Weighted mean aggregation
            neighbor_sum = ops.matmul(adjacency_weights, neighbor_features)
            degree = ops.sum(adjacency_weights, axis=-1, keepdims=True)
            degree = ops.maximum(degree, keras.backend.epsilon())
            return neighbor_sum / degree

        elif self.aggregator == 'max':
            # Max aggregation with masking
            expanded_neighbors = ops.expand_dims(neighbor_features, axis=1)  # (batch, 1, num_neighbors, dim)
            expanded_adj = ops.expand_dims(adjacency_weights, axis=-1)  # (batch, num_nodes, num_neighbors, 1)
            masked_features = expanded_neighbors * expanded_adj
            # Set masked positions to large negative value for max operation
            mask = ops.expand_dims(adjacency_weights, axis=-1) > 0
            masked_features = ops.where(mask, masked_features, -1e9)
            return ops.max(masked_features, axis=2)

        elif self.aggregator == 'sum':
            # Sum aggregation
            return ops.matmul(adjacency_weights, neighbor_features)

        elif self.aggregator == 'attention':
            # Attention-based aggregation
            num_nodes = ops.shape(node_features)[1]
            num_neighbors = ops.shape(neighbor_features)[1]

            # Expand for pairwise attention computation
            node_exp = ops.expand_dims(node_features, axis=2)  # (batch, nodes, 1, dim)
            neighbor_exp = ops.expand_dims(neighbor_features, axis=1)  # (batch, 1, neighbors, dim)

            # Broadcast to create all pairs
            node_broadcast = ops.broadcast_to(node_exp, (ops.shape(node_features)[0], num_nodes, num_neighbors,
                                                         ops.shape(node_features)[2]))
            neighbor_broadcast = ops.broadcast_to(neighbor_exp,
                                                  (ops.shape(neighbor_features)[0], num_nodes, num_neighbors,
                                                   ops.shape(neighbor_features)[2]))

            # Concatenate and compute attention
            concat_features = ops.concatenate([node_broadcast, neighbor_broadcast], axis=-1)
            attention_scores = ops.squeeze(ops.matmul(concat_features, self.W_att), axis=-1)

            # Apply adjacency mask and softmax
            masked_scores = ops.where(adjacency_weights > 0, attention_scores, -1e9)
            attention_weights = ops.softmax(masked_scores, axis=-1)

            return ops.matmul(attention_weights, neighbor_features)

        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

    def call(self, inputs, training=None):
        """Forward pass implementing CROWN's bipartite GraphSAGE.
        
        Args:
            inputs: Tuple of (user_features, news_features, adjacency_matrix)
                - user_features: (batch_size, num_users, user_dim)
                - news_features: (batch_size, num_news, news_dim)
                - adjacency_matrix: (batch_size, num_users + num_news, num_users + num_news)
            training: Whether in training mode
            
        Returns:
            Tuple of (updated_user_features, updated_news_features)
        """
        user_features, news_features, adjacency_matrix = inputs

        batch_size = ops.shape(user_features)[0]
        num_users = ops.shape(user_features)[1]
        num_news = ops.shape(news_features)[1]

        # Extract bipartite adjacency submatrices
        # User-to-news connections: adjacency_matrix[:, :num_users, num_users:]
        user_to_news = adjacency_matrix[:, :num_users, num_users:]
        # News-to-user connections: adjacency_matrix[:, num_users:, :num_users]  
        news_to_user = adjacency_matrix[:, num_users:, :num_users]

        # Step 1: Update user embeddings by aggregating from connected news
        news_for_users = self.aggregate_neighbors(user_features, news_features, user_to_news)
        user_self = ops.matmul(user_features, self.W_user_self)
        user_neigh = ops.matmul(news_for_users, self.W_user_neigh)
        updated_users = self.activation(user_self + user_neigh + self.b_user)
        updated_users = self.dropout(updated_users, training=training)

        # Step 2: Update news embeddings by aggregating from connected users
        users_for_news = self.aggregate_neighbors(news_features, user_features, news_to_user)
        news_self = ops.matmul(news_features, self.W_news_self)
        news_neigh = ops.matmul(users_for_news, self.W_news_neigh)
        updated_news = self.activation(news_self + news_neigh + self.b_news)
        updated_news = self.dropout(updated_news, training=training)

        # Optional L2 normalization as per GraphSAGE paper
        if self.normalize:
            # L2 normalize manually since ops.nn.l2_normalize doesn't exist
            user_norm = ops.sqrt(ops.sum(ops.square(updated_users), axis=-1, keepdims=True))
            updated_users = updated_users / (user_norm + keras.backend.epsilon())
            
            news_norm = ops.sqrt(ops.sum(ops.square(updated_news), axis=-1, keepdims=True))
            updated_news = updated_news / (news_norm + keras.backend.epsilon())

        return updated_users, updated_news

    def compute_output_shape(self, input_shape):
        user_shape, news_shape, _ = input_shape
        return (
            (user_shape[0], user_shape[1], self.units),
            (news_shape[0], news_shape[1], self.units)
        )


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


class GraphAttentionLayer(layers.Layer):
    """Graph Attention Network (GAT) layer implementation for CROWN paper.
    
    This layer implements GAT specifically for the bipartite user-news graph
    used in the CROWN model, with attention-based mutual updates between 
    user and news embeddings.
    
    Args:
        units: Output dimension for both user and news embeddings
        num_heads: Number of attention heads
        dropout_rate: Dropout rate
        activation: Activation function
        seed: Random seed
        use_bias: Whether to use bias in linear transformations
        alpha: LeakyReLU negative slope for attention mechanism
        concat_heads: Whether to concatenate or average multi-head outputs
    """

    def __init__(
            self,
            units,
            num_heads=1,
            dropout_rate=0.0,
            activation='relu',
            seed=42,
            use_bias=True,
            alpha=0.2,
            concat_heads=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = keras.activations.get(activation)
        self.seed = seed
        self.use_bias = use_bias
        self.alpha = alpha  # LeakyReLU negative slope
        self.concat_heads = concat_heads

        # Compute head dimension
        if concat_heads:
            assert units % num_heads == 0, "units must be divisible by num_heads when concat_heads=True"
            self.head_dim = units // num_heads
        else:
            self.head_dim = units

    def build(self, input_shape):
        """Build the GAT layer weights.
        
        Expected input shapes:
        - user_features: (batch_size, num_users, user_feature_dim)
        - news_features: (batch_size, num_news, news_feature_dim)
        - adjacency_matrix: (batch_size, num_users + num_news, num_users + num_news)
        """
        if len(input_shape) != 3:
            raise ValueError("GraphAttentionLayer expects 3 inputs: (user_features, news_features, adjacency_matrix)")

        user_features_shape, news_features_shape, _ = input_shape
        user_dim = user_features_shape[-1]
        news_dim = news_features_shape[-1]

        # User transformation weights for each attention head
        self.W_user = self.add_weight(
            name='W_user',
            shape=(self.num_heads, user_dim, self.head_dim),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True
        )

        # News transformation weights for each attention head
        self.W_news = self.add_weight(
            name='W_news',
            shape=(self.num_heads, news_dim, self.head_dim),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True
        )

        # Attention mechanism weights
        # For user-news attention (user as query, news as key/value)
        self.a_user_news = self.add_weight(
            name='a_user_news',
            shape=(self.num_heads, self.head_dim * 2, 1),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True
        )

        # For news-user attention (news as query, user as key/value)
        self.a_news_user = self.add_weight(
            name='a_news_user',
            shape=(self.num_heads, self.head_dim * 2, 1),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True
        )

        # Bias terms
        if self.use_bias:
            self.b_user = self.add_weight(
                name='b_user',
                shape=(self.units if self.concat_heads else self.head_dim,),
                initializer=keras.initializers.Zeros(),
                trainable=True
            )

            self.b_news = self.add_weight(
                name='b_news',
                shape=(self.units if self.concat_heads else self.head_dim,),
                initializer=keras.initializers.Zeros(),
                trainable=True
            )

        self.dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        super().build(input_shape)

    def compute_attention(self, queries, keys, attention_weights, adjacency_mask, training=None):
        """Compute attention scores using GAT mechanism.
        
        Args:
            queries: Query features (batch_size, num_heads, num_queries, head_dim)
            keys: Key features (batch_size, num_heads, num_keys, head_dim)
            attention_weights: Attention parameter (num_heads, head_dim * 2, 1)
            adjacency_mask: Binary mask for valid connections (batch_size, num_queries, num_keys)
            training: Whether in training mode
            
        Returns:
            Attention coefficients (batch_size, num_heads, num_queries, num_keys)
        """
        batch_size = ops.shape(queries)[0]
        num_queries = ops.shape(queries)[2]
        num_keys = ops.shape(keys)[2]

        # Expand queries and keys for pairwise computation
        queries_exp = ops.expand_dims(queries, axis=3)  # (batch, heads, queries, 1, head_dim)
        keys_exp = ops.expand_dims(keys, axis=2)  # (batch, heads, 1, keys, head_dim)

        # Broadcast to create all query-key pairs
        queries_broadcast = ops.broadcast_to(
            queries_exp, (batch_size, self.num_heads, num_queries, num_keys, self.head_dim)
        )
        keys_broadcast = ops.broadcast_to(
            keys_exp, (batch_size, self.num_heads, num_queries, num_keys, self.head_dim)
        )

        # Concatenate queries and keys
        concat_qk = ops.concatenate([queries_broadcast, keys_broadcast], axis=-1)

        # Compute attention scores for each head
        attention_scores = ops.squeeze(
            ops.matmul(concat_qk, ops.expand_dims(attention_weights, axis=0)), axis=-1
        )  # (batch, heads, queries, keys)

        # Apply LeakyReLU
        attention_scores = ops.leaky_relu(attention_scores, negative_slope=self.alpha)

        # Apply adjacency mask (set invalid connections to large negative value)
        adjacency_mask_exp = ops.expand_dims(adjacency_mask, axis=1)  # (batch, 1, queries, keys)
        masked_scores = ops.where(
            adjacency_mask_exp > 0, attention_scores, -1e9
        )

        # Apply softmax to get attention coefficients
        attention_coeffs = ops.softmax(masked_scores, axis=-1)

        # Apply dropout to attention coefficients
        attention_coeffs = self.dropout(attention_coeffs, training=training)

        return attention_coeffs

    def call(self, inputs, training=None):
        """Forward pass implementing CROWN's bipartite GAT.
        
        Args:
            inputs: Tuple of (user_features, news_features, adjacency_matrix)
                - user_features: (batch_size, num_users, user_dim)
                - news_features: (batch_size, num_news, news_dim)
                - adjacency_matrix: (batch_size, num_users + num_news, num_users + num_news)
            training: Whether in training mode
            
        Returns:
            Tuple of (updated_user_features, updated_news_features)
        """
        user_features, news_features, adjacency_matrix = inputs

        batch_size = ops.shape(user_features)[0]
        num_users = ops.shape(user_features)[1]
        num_news = ops.shape(news_features)[1]

        # Extract bipartite adjacency submatrices
        user_to_news = adjacency_matrix[:, :num_users, num_users:]  # (batch, users, news)
        news_to_user = adjacency_matrix[:, num_users:, :num_users]  # (batch, news, users)

        # Transform features for each attention head
        # User transformations: (batch, num_heads, users, head_dim)
        user_transformed = ops.einsum('bud,hdk->bhuk', user_features, self.W_user)
        # News transformations: (batch, num_heads, news, head_dim)
        news_transformed = ops.einsum('bnd,hdk->bhnk', news_features, self.W_news)

        # Step 1: Update user embeddings using attention over connected news
        user_news_attention = self.compute_attention(
            user_transformed, news_transformed, self.a_user_news, user_to_news, training=training
        )  # (batch, heads, users, news)

        # Aggregate news features for users
        user_updates = ops.einsum('bhun,bhnk->bhuk', user_news_attention, news_transformed)

        # Step 2: Update news embeddings using attention over connected users  
        news_user_attention = self.compute_attention(
            news_transformed, user_transformed, self.a_news_user, news_to_user, training=training
        )  # (batch, heads, news, users)

        # Aggregate user features for news
        news_updates = ops.einsum('bhnu,bhuk->bhnk', news_user_attention, user_transformed)

        # Handle multi-head outputs
        if self.concat_heads:
            # Concatenate heads: (batch, nodes, units)
            updated_users = ops.reshape(user_updates, (batch_size, num_users, self.units))
            updated_news = ops.reshape(news_updates, (batch_size, num_news, self.units))
        else:
            # Average heads: (batch, nodes, head_dim)
            updated_users = ops.mean(user_updates, axis=1)
            updated_news = ops.mean(news_updates, axis=1)

        # Add bias and apply activation
        if self.use_bias:
            updated_users = updated_users + self.b_user
            updated_news = updated_news + self.b_news

        updated_users = self.activation(updated_users)
        updated_news = self.activation(updated_news)

        # Apply dropout
        updated_users = self.dropout(updated_users, training=training)
        updated_news = self.dropout(updated_news, training=training)

        return updated_users, updated_news

    def compute_output_shape(self, input_shape):
        user_shape, news_shape, _ = input_shape
        output_dim = self.units if self.concat_heads else self.head_dim
        return (
            (user_shape[0], user_shape[1], output_dim),
            (news_shape[0], news_shape[1], output_dim)
        )

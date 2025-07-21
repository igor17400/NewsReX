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


class MAB(layers.Layer):
    """Multihead Attention Block as defined in the original CROWN paper.

    This layer implements a multihead attention mechanism with residual connections
    and layer normalization, followed by a feedforward network.
    
    Based on the original PyTorch implementation but using Keras's MultiHeadAttention:
    - Uses Keras MultiHeadAttention for the attention mechanism
    - Applies layer norm and feedforward with residual connections
    - Follows the same structure as the original MAB

    Args:
        embed_dim (int): The embedding dimension (dim_V in original).
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimension of the feedforward network.
        dropout_rate (float): Dropout rate for regularization.
        ln (bool): Whether to use layer normalization.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, ln=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.ln = ln

        # Multihead attention layer (using Keras built-in)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads, 
            dropout=dropout_rate
        )

        # Layer normalization layers (optional)
        if ln:
            self.ln0 = layers.LayerNormalization(epsilon=1e-6)
            self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        else:
            self.ln0 = None
            self.ln1 = None

        # Feedforward network (following original structure)
        self.fc_o = layers.Dense(embed_dim, use_bias=True)

    def call(self, Q, K, training=None):
        """
        Forward pass of the MAB layer.

        Args:
            Q: Query tensor.
            K: Key tensor (also used as value).
            training: Whether in training mode.

        Returns:
            Output tensor after multihead attention and feedforward processing.
        """
        # Multihead attention with residual connection
        attn_output = self.mha(Q, K, K, training=training)
        O = Q + attn_output  # Residual connection
        
        # Layer normalization (if enabled)
        if self.ln0 is not None:
            O = self.ln0(O)
        
        # Feedforward network with residual connection
        ff_output = self.fc_o(O)
        O = O + ops.relu(ff_output)
        
        # Layer normalization (if enabled)
        if self.ln1 is not None:
            O = self.ln1(O)
        
        return O

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "ln": self.ln,
            }
        )
        return config


class IntentDisentanglementLayer(layers.Layer):
    """K-intent disentanglement layer for CROWN model.

    This layer applies k different linear transformations to learn k different
    intent representations from the input news embedding.

    Args:
        intent_num (int): Number of intents (k) to disentangle.
        intent_embedding_dim (int): Dimension of each intent embedding.
        dropout_rate (float): Dropout rate for regularization.
    """

    def __init__(self, intent_num, intent_embedding_dim, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.intent_num = intent_num
        self.intent_embedding_dim = intent_embedding_dim
        self.dropout_rate = dropout_rate

        # K different linear layers for intent disentanglement
        self.intent_layers = [
            layers.Dense(intent_embedding_dim, activation="relu", name=f"intent_layer_{i}")
            for i in range(intent_num)
        ]

    def call(self, news_embedding):
        """Apply k-FC layer for k-intent disentanglement."""
        k_intent_embeddings = []

        for i in range(self.intent_num):
            intent_embedding = self.intent_layers[i](news_embedding)
            intent_embedding = ops.expand_dims(intent_embedding, axis=1)
            k_intent_embeddings.append(intent_embedding)

        # Concatenate along axis 1
        k_intent_embeddings = ops.concatenate(k_intent_embeddings, axis=1)
        return k_intent_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intent_num": self.intent_num,
                "intent_embedding_dim": self.intent_embedding_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class CategoryPredictor(layers.Layer):
    """Category predictor for auxiliary loss in CROWN model.

    This layer predicts the category of a news article from its intent embedding
    and computes the categorical crossentropy loss.

    Args:
        intent_embedding_dim (int): Dimension of the intent embedding.
        category_num (int): Number of categories to predict.
    """

    def __init__(self, intent_embedding_dim, category_num, **kwargs):
        super().__init__(**kwargs)
        self.intent_embedding_dim = intent_embedding_dim
        self.category_num = category_num

        self.category_classifier = layers.Dense(category_num, activation="softmax")

    def call(self, intent_embedding, target_category):
        """Predict category and compute loss."""
        predicted_category = self.category_classifier(intent_embedding)

        # Compute categorical crossentropy loss
        loss = keras.losses.sparse_categorical_crossentropy(
            target_category, predicted_category, from_logits=False
        )

        return ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intent_embedding_dim": self.intent_embedding_dim,
                "category_num": self.category_num,
            }
        )
        return config


class CustomGATLayer(layers.Layer):
    """Custom Graph Attention Network layer for efficient batch processing.
    
    This implementation is based on the GAT paper but optimized for the CROWN model:
    - Supports batch processing without loops
    - Uses sparse adjacency matrix operations
    - Follows the original PyTorch CROWN attention mechanism
    
    Args:
        output_dim (int): Output feature dimension
        num_heads (int): Number of attention heads
        dropout_rate (float): Dropout rate for attention and features
        use_bias (bool): Whether to use bias in linear transformations
        activation (str): Activation function ('relu', 'elu', etc.)
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias weights
        concat_heads (bool): Whether to concatenate or average attention heads
    """
    
    def __init__(
        self,
        output_dim,
        num_heads=8,
        dropout_rate=0.2,
        use_bias=True,
        activation="relu",
        kernel_regularizer=None,
        bias_regularizer=None,
        concat_heads=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.concat_heads = concat_heads
        
        # Calculate dimensions
        if concat_heads:
            self.head_dim = output_dim // num_heads
            self.final_output_dim = output_dim
        else:
            self.head_dim = output_dim
            self.final_output_dim = output_dim
            
        # Dropout layers
        self.dropout = layers.Dropout(dropout_rate)
        self.attention_dropout = layers.Dropout(dropout_rate)
        
        # Activation function
        if activation == "relu":
            self.activation_fn = ops.relu
        elif activation == "elu":
            self.activation_fn = ops.elu
        else:
            self.activation_fn = ops.relu
    
    def build(self, input_shape):
        """Build the layer weights."""
        node_features_shape = input_shape[0]
        input_dim = node_features_shape[-1]
        
        # Weight matrices for each attention head
        self.W = self.add_weight(
            name="W",
            shape=(self.num_heads, input_dim, self.head_dim),
            initializer="glorot_uniform",
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        
        # Attention mechanism weights (a vector for each head)
        self.a = self.add_weight(
            name="a",
            shape=(self.num_heads, 2 * self.head_dim),
            initializer="glorot_uniform",
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        
        # Bias weights
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.final_output_dim,),
                initializer="zeros",
                regularizer=self.bias_regularizer,
                trainable=True,
            )
        else:
            self.bias = None
            
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Forward pass of the GAT layer.
        
        Args:
            inputs: List of [node_features, adjacency_matrix]
                - node_features: (batch_size, num_nodes, input_dim)
                - adjacency_matrix: (batch_size, num_nodes, num_nodes)
            training: Whether in training mode
            
        Returns:
            Output features: (batch_size, num_nodes, output_dim)
        """
        node_features, adjacency_matrix = inputs
        batch_size = ops.shape(node_features)[0]
        num_nodes = ops.shape(node_features)[1]
        
        # Apply dropout to input features
        if training:
            node_features = self.dropout(node_features, training=training)
        
        # Linear transformation for each head
        # node_features: (batch_size, num_nodes, input_dim)
        # W: (num_heads, input_dim, head_dim)
        # Result: (batch_size, num_heads, num_nodes, head_dim)
        node_features_transformed = ops.einsum("bni,hij->bhnj", node_features, self.W)
        
        # Compute attention coefficients
        # For each head, compute attention between all node pairs
        attention_outputs = []
        
        for head in range(self.num_heads):
            # Get features for this head: (batch_size, num_nodes, head_dim)
            features_head = node_features_transformed[:, head, :, :]
            
            # Compute attention coefficients
            # Create all pairs: (batch_size, num_nodes, num_nodes, 2*head_dim)
            features_i = ops.expand_dims(features_head, axis=2)  # (batch_size, num_nodes, 1, head_dim)
            features_j = ops.expand_dims(features_head, axis=1)  # (batch_size, 1, num_nodes, head_dim)
            
            features_i = ops.tile(features_i, (1, 1, num_nodes, 1))  # (batch_size, num_nodes, num_nodes, head_dim)
            features_j = ops.tile(features_j, (1, num_nodes, 1, 1))  # (batch_size, num_nodes, num_nodes, head_dim)
            
            # Concatenate features: (batch_size, num_nodes, num_nodes, 2*head_dim)
            attention_input = ops.concatenate([features_i, features_j], axis=-1)
            
            # Compute attention scores: (batch_size, num_nodes, num_nodes)
            attention_scores = ops.einsum("bijk,k->bij", attention_input, self.a[head])
            
            # Apply LeakyReLU
            attention_scores = ops.maximum(attention_scores, 0.2 * attention_scores)
            
            # Mask attention scores using adjacency matrix
            # Set attention to very negative value where there's no edge
            mask = ops.cast(adjacency_matrix, dtype=self.compute_dtype)
            masked_attention = attention_scores * mask + (1.0 - mask) * (-1e9)
            
            # Apply softmax to get attention weights
            attention_weights = ops.softmax(masked_attention, axis=-1)
            
            # Apply attention dropout
            if training:
                attention_weights = self.attention_dropout(attention_weights, training=training)
            
            # Apply attention to features
            # attention_weights: (batch_size, num_nodes, num_nodes)
            # features_head: (batch_size, num_nodes, head_dim)
            # Result: (batch_size, num_nodes, head_dim)
            attended_features = ops.einsum("bij,bjk->bik", attention_weights, features_head)
            attention_outputs.append(attended_features)
        
        # Combine attention heads
        if self.concat_heads:
            # Concatenate all heads
            output = ops.concatenate(attention_outputs, axis=-1)
        else:
            # Average all heads
            output = ops.mean(ops.stack(attention_outputs, axis=0), axis=0)
        
        # Apply bias
        if self.use_bias:
            output = output + self.bias
        
        # Apply activation
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "concat_heads": self.concat_heads,
        })
        return config


class BipartiteGraphCreator(layers.Layer):
    """Creates bipartite graph adjacency matrix for user-news interactions.
    
    This layer creates a simple bipartite graph structure matching the original 
    PyTorch implementation, where users are connected to all their history items.
    
    Args:
        max_history_length (int): Maximum number of history items per user
    """
    
    def __init__(self, max_history_length, **kwargs):
        super().__init__(**kwargs)
        self.max_history_length = max_history_length
    
    def call(self, history_mask):
        """
        Create bipartite adjacency matrix from history mask.
        
        Args:
            history_mask: (batch_size, max_history_length) - boolean mask for valid history
            
        Returns:
            adjacency_matrix: (batch_size, max_history_length, max_history_length)
        """
        batch_size = ops.shape(history_mask)[0]
        
        # Create identity matrix for self-connections
        identity = ops.eye(self.max_history_length, dtype=self.compute_dtype)
        identity = ops.expand_dims(identity, axis=0)
        identity = ops.tile(identity, (batch_size, 1, 1))
        
        # Create mask for valid connections
        # Each valid news item connects to all other valid news items
        history_mask_expanded = ops.expand_dims(history_mask, axis=1)  # (batch_size, 1, max_history_length)
        history_mask_transposed = ops.expand_dims(history_mask, axis=2)  # (batch_size, max_history_length, 1)
        
        # Outer product to create adjacency matrix
        adjacency = ops.cast(history_mask_transposed, self.compute_dtype) * ops.cast(history_mask_expanded, self.compute_dtype)
        
        # Add self-connections
        adjacency = adjacency + identity
        
        # Ensure no connections beyond valid history length
        adjacency = adjacency * ops.expand_dims(ops.cast(history_mask, self.compute_dtype), axis=1)
        adjacency = adjacency * ops.expand_dims(ops.cast(history_mask, self.compute_dtype), axis=2)
        
        return adjacency
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_history_length": self.max_history_length,
        })
        return config


class UserAttentionLayer(layers.Layer):
    """User attention mechanism for CROWN model.
    
    This layer implements the attention mechanism to compute user representation
    from GAT features, matching the original PyTorch implementation.
    
    Args:
        attention_dim (int): Dimension of attention mechanism
        news_embedding_dim (int): Dimension of news embeddings
    """
    
    def __init__(self, attention_dim, news_embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.news_embedding_dim = news_embedding_dim
        
    def build(self, input_shape):
        # Key matrix
        self.K = layers.Dense(
            self.attention_dim, 
            use_bias=False, 
            name="attention_key"
        )
        
        # User node embedding for query (matching original implementation)
        self.user_node_embedding = self.add_weight(
            name="user_node_embedding",
            shape=(1, self.news_embedding_dim),
            initializer="zeros",
            trainable=True
        )
        
        # Query matrix - build with explicit input shape
        self.Q = layers.Dense(
            self.attention_dim, 
            use_bias=True, 
            name="attention_query"
        )
        # Build the Dense layer with the correct input shape
        self.Q.build((None, self.news_embedding_dim))
        
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Compute user representation using attention mechanism.
        
        Args:
            inputs: List of [gat_features, history_mask]
                - gat_features: (batch_size, max_history_length, news_embedding_dim)
                - history_mask: (batch_size, max_history_length)
                
        Returns:
            user_representation: (batch_size, news_embedding_dim)
        """
        gat_features, history_mask = inputs
        batch_size = ops.shape(gat_features)[0]
        
        # Key matrix
        K = self.K(gat_features)  # (batch_size, max_history_length, attention_dim)
        
        # Expand user embedding for batch
        user_embedding_batch = ops.tile(self.user_node_embedding, (batch_size, 1))
        
        # Query matrix
        Q = self.Q(user_embedding_batch)  # (batch_size, attention_dim)
        Q = ops.expand_dims(Q, axis=-1)  # (batch_size, attention_dim, 1)

        # Compute attention scores
        attention_scores = ops.squeeze(
            ops.matmul(K, Q), 
            axis=-1
        ) / ops.sqrt(ops.cast(self.attention_dim, self.compute_dtype))
        
        # Apply mask to attention scores
        mask_value = -1e9
        masked_scores = ops.where(
            history_mask, 
            attention_scores, 
            mask_value
        )
        
        # Softmax attention weights
        attention_weights = ops.softmax(masked_scores, axis=1)
        
        # Apply attention to get user representation
        user_representation = ops.sum(
            ops.expand_dims(attention_weights, -1) * gat_features, 
            axis=1
        )
        
        return user_representation
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "attention_dim": self.attention_dim,
            "news_embedding_dim": self.news_embedding_dim,
        })
        return config

from typing import Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

import keras
from keras import layers, ops

from .layers import AdditiveAttentionLayer, GraphSAGELayer, InducedSetAttentionBlock
from .base import BaseModel


@dataclass
class CROWNConfig:
    """Configuration class for CROWN model parameters."""
    # Common parameters
    embedding_size: int = 300
    dropout_rate: float = 0.2
    seed: int = 42

    # Model dimensions
    intent_embedding_dim: int = 200
    category_embedding_dim: int = 100
    subcategory_embedding_dim: int = 100
    attention_dim: int = 200

    # Intent disentanglement
    intent_num: int = 4  # Number of intents (k)
    alpha: float = 0.5  # Weight for auxiliary loss

    # Transformer parameters
    num_heads: int = 8
    head_dim: int = 40  # 8 * 40 = 320 > 300d embeddings
    feedforward_dim: int = 512
    num_layers: int = 2

    # ISAB parameters
    isab_num_heads: int = 4
    isab_num_inducing_points: int = 4

    # GraphSAGE parameters
    graph_hidden_dim: int = 300
    graph_num_layers: int = 1

    # Input parameters
    max_title_length: int = 50
    max_abstract_length: int = 100
    max_history_length: int = 50
    max_impressions_length: int = 5

    # Training parameters
    process_user_id: bool = False


class PositionalEncoding(layers.Layer):
    """Positional encoding layer for transformer models."""

    def __init__(self, dropout_rate=0.1, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.dropout = layers.Dropout(dropout_rate)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]

        # Create positional encoding matrix using numpy for initialization
        position = np.arange(0, self.max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) *
                          (-np.log(10000.0) / d_model))

        pe = np.zeros((self.max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Add batch dimension
        pe = pe.reshape(1, self.max_len, d_model)

        self.pe = self.add_weight(
            name='pe',
            shape=(1, self.max_len, d_model),
            initializer=keras.initializers.Constant(pe),
            trainable=False
        )

        super().build(input_shape)

    def call(self, x, training=None):
        """Add positional encoding to input embeddings."""
        seq_len = ops.shape(x)[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, training=training)


class CategoryPredictor(layers.Layer):
    """Category prediction layer for auxiliary task."""

    def __init__(self, category_num, **kwargs):
        super().__init__(**kwargs)
        self.category_num = category_num

    def build(self, input_shape):
        self.fc = layers.Dense(
            self.category_num,
            kernel_initializer='glorot_uniform',
            name='category_predictor_dense'
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Predict category from intent embeddings.
        
        Args:
            inputs: Intent embeddings (batch_size * news_num, intent_embedding_dim)
            
        Returns:
            Category logits (batch_size * news_num, category_num)
        """
        return self.fc(inputs)


class CROWNNewsEncoder(keras.Model):
    """News encoder component for CROWN model.
    
    Implements category-guided intent disentanglement and consistency-based
    news representation.
    """

    def __init__(self, config: CROWNConfig, embedding_layer: layers.Embedding,
                 category_embedding: layers.Embedding, subcategory_embedding: layers.Embedding,
                 name: str = "crown_news_encoder"):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer
        self.category_embedding = category_embedding
        self.subcategory_embedding = subcategory_embedding

        # News embedding dimension calculation
        self.news_embedding_dim = (config.intent_embedding_dim * 2 +
                                   config.category_embedding_dim +
                                   config.subcategory_embedding_dim)

        # Dropout layer
        self.dropout = layers.Dropout(config.dropout_rate, seed=config.seed)

        # Positional encoding
        self.title_pos_encoder = PositionalEncoding(config.dropout_rate, config.max_title_length)
        self.body_pos_encoder = PositionalEncoding(config.dropout_rate, config.max_abstract_length)

        # Transformer encoders
        self.title_transformer = self._build_transformer_encoder("title")
        self.body_transformer = self._build_transformer_encoder("body")

        # ISAB encoder (optional, currently commented out in original)
        self.isab = InducedSetAttentionBlock(
            dim_out=config.embedding_size,
            num_heads=config.isab_num_heads,
            num_inducing_points=config.isab_num_inducing_points,
            use_layer_norm=True,
            seed=config.seed
        )

        # Category affine transformation
        self.category_affine = layers.Dense(
            config.category_embedding_dim,
            kernel_initializer='glorot_uniform',
            name='category_affine'
        )

        # Intent layers for k-intent disentanglement
        self.intent_layers = []
        for i in range(config.intent_num):
            self.intent_layers.append(
                layers.Dense(
                    config.intent_embedding_dim,
                    activation='relu',
                    kernel_initializer='glorot_uniform',
                    name=f'intent_layer_{i}'
                )
            )

        # Intent attention layers
        self.title_intent_attention = AdditiveAttentionLayer(
            query_vec_dim=config.attention_dim,
            seed=config.seed,
            name='title_intent_attention'
        )
        self.body_intent_attention = AdditiveAttentionLayer(
            query_vec_dim=config.attention_dim,
            seed=config.seed,
            name='body_intent_attention'
        )

        # Category predictor for auxiliary task - will be set in _create_components
        self.category_predictor = None

        # For auxiliary loss tracking
        self.auxiliary_loss = None

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the news encoder.
        
        Args:
            input_shape: Input shape (batch_size, title_length + abstract_length + 2)
            
        Returns:
            Output shape (batch_size, news_embedding_dim)
        """
        return (input_shape[0], self.news_embedding_dim)

    def _build_transformer_encoder(self, name_prefix):
        """Build a transformer encoder stack using custom implementation."""

        class TransformerBlock(layers.Layer):
            def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, **kwargs):
                super().__init__(**kwargs)
                self.att = layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embed_dim // num_heads,
                    dropout=dropout_rate
                )
                self.ffn = keras.Sequential([
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ])
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = layers.Dropout(dropout_rate)
                self.dropout2 = layers.Dropout(dropout_rate)

            def call(self, inputs, training=None, mask=None):
                attn_output = self.att(inputs, inputs, training=training)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)

        # Create transformer encoder stack
        transformer_blocks = []
        for i in range(self.config.num_layers):
            transformer_blocks.append(
                TransformerBlock(
                    embed_dim=self.config.embedding_size,
                    num_heads=self.config.num_heads,
                    ff_dim=self.config.feedforward_dim,
                    dropout_rate=self.config.dropout_rate,
                    name=f'{name_prefix}_transformer_block_{i}'
                )
            )

        return keras.Sequential(transformer_blocks, name=f'{name_prefix}_transformer')

    def k_intent_disentangle(self, news_embedding):
        """Apply k-FC layers for k-intent disentanglement.
        
        Args:
            news_embedding: (batch_size * news_num, embedding_dim)
            
        Returns:
            k_intent_embeddings: (batch_size * news_num, k, intent_embedding_dim)
        """
        k_intent_embeddings = []

        for i in range(self.config.intent_num):
            # Apply different linear transformation for each intent
            intent_embedding = self.intent_layers[i](news_embedding)
            # Expand dimension
            intent_embedding_exp = ops.expand_dims(intent_embedding, axis=1)
            k_intent_embeddings.append(intent_embedding_exp)

        # Concatenate along the intent dimension
        k_intent_embeddings = ops.concatenate(k_intent_embeddings, axis=1)

        return k_intent_embeddings

    def similarity_compute(self, title, body):
        """Compute title-body similarity using cosine similarity.
        
        Args:
            title: (batch_size * news_num, intent_embedding_dim)
            body: (batch_size * news_num, intent_embedding_dim)
            
        Returns:
            similarity: (batch_size * news_num,)
        """
        # Normalize vectors
        title_norm = ops.sqrt(ops.sum(ops.square(title), axis=1, keepdims=True))
        body_norm = ops.sqrt(ops.sum(ops.square(body), axis=1, keepdims=True))

        title_normalized = title / (title_norm + keras.backend.epsilon())
        body_normalized = body / (body_norm + keras.backend.epsilon())

        # Compute cosine similarity
        cosine_similarity = ops.sum(title_normalized * body_normalized, axis=1)

        # Scale to [0, 1]
        title_body_similarity = (cosine_similarity + 1) / 2.0

        return title_body_similarity

    def call(self, inputs, training=None):
        """Forward pass for news encoding.
        
        Args:
            inputs: Concatenated tensor containing [title_tokens, abstract_tokens, category_id, subcategory_id]
                   Shape: (batch_size, title_length + abstract_length + 2)
            training: Whether in training mode
            
        Returns:
            News representations (batch_size, news_embedding_dim)
        """
        # Split the concatenated input: [title, abstract, category, subcategory] 
        # Calculate actual dimensions from input shape
        total_features = ops.shape(inputs)[-1]
        actual_abstract_length = total_features - self.config.max_title_length - 2  # 2 for category + subcategory
        
        title_text = inputs[:, :self.config.max_title_length]
        content_text_raw = inputs[:, self.config.max_title_length:self.config.max_title_length + actual_abstract_length]
        category = inputs[:, self.config.max_title_length + actual_abstract_length:self.config.max_title_length + actual_abstract_length + 1]
        subcategory = inputs[:, self.config.max_title_length + actual_abstract_length + 1:]

        # Pad or truncate content_text to match config.max_abstract_length
        if actual_abstract_length < self.config.max_abstract_length:
            # Pad with zeros
            padding_size = self.config.max_abstract_length - actual_abstract_length
            batch_size = ops.shape(content_text_raw)[0]
            padding = ops.zeros((batch_size, padding_size), dtype=content_text_raw.dtype)
            content_text = ops.concatenate([content_text_raw, padding], axis=1)
        elif actual_abstract_length > self.config.max_abstract_length:
            # Truncate
            content_text = content_text_raw[:, :self.config.max_abstract_length]
        else:
            content_text = content_text_raw

        # Squeeze category and subcategory to remove extra dimension
        category = ops.squeeze(category, axis=-1)
        subcategory = ops.squeeze(subcategory, axis=-1)

        # Cast to int32 if needed
        category = ops.cast(category, dtype='int32')
        subcategory = ops.cast(subcategory, dtype='int32')

        # Word embeddings
        title_w = self.dropout(self.embedding_layer(title_text), training=training)
        body_w = self.dropout(self.embedding_layer(content_text), training=training)

        # Positional encoding
        title_p = self.title_pos_encoder(title_w, training=training)
        body_p = self.body_pos_encoder(body_w, training=training)

        # Transformer encoding
        title_t = self.title_transformer(title_p, training=training)
        body_t = self.body_transformer(body_p, training=training)

        # Mean pooling
        title_embedding = ops.mean(title_t, axis=1)
        body_embedding = ops.mean(body_t, axis=1)

        # Category-aware intent disentanglement
        cat_embed = self.category_embedding(category)
        subcat_embed = self.subcategory_embedding(subcategory)
        category_concat = ops.concatenate([cat_embed, subcat_embed], axis=1)
        category_representation = self.category_affine(category_concat)

        # Concatenate with category information
        category_aware_title = ops.concatenate([title_embedding, category_representation], axis=1)
        category_aware_body = ops.concatenate([body_embedding, category_representation], axis=1)

        # K-intent disentanglement
        title_k_intents = self.k_intent_disentangle(category_aware_title)
        body_k_intents = self.k_intent_disentangle(category_aware_body)

        # Intent-based attention
        title_intent_embedding = self.title_intent_attention(title_k_intents)
        body_intent_embedding = self.body_intent_attention(body_k_intents)

        # Category predictor (auxiliary task)
        if training and self.category_predictor is not None:
            category_logits = self.category_predictor(title_intent_embedding)
            category_one_hot = ops.one_hot(category, self.category_predictor.category_num)
            category_loss = ops.categorical_crossentropy(
                category_one_hot,
                category_logits,
                from_logits=True
            )
            self.auxiliary_loss = ops.mean(category_loss) * self.config.alpha

        # Title-body similarity computation
        similarity = self.similarity_compute(title_intent_embedding, body_intent_embedding)
        similarity_expanded = ops.expand_dims(similarity, axis=1)

        # Consistency-based representation
        news_representation = ops.concatenate([
            title_intent_embedding,
            similarity_expanded * body_intent_embedding
        ], axis=1)

        # Add category and subcategory embeddings
        news_representation = ops.concatenate([
            news_representation,
            self.dropout(cat_embed, training=training),
            self.dropout(subcat_embed, training=training)
        ], axis=1)

        return news_representation


class CROWNUserEncoder(keras.Model):
    """User encoder component for CROWN model.
    
    Implements GNN-enhanced hybrid user representation using GraphSAGE.
    """

    def __init__(self, config: CROWNConfig, news_encoder: CROWNNewsEncoder,
                 name: str = "crown_user_encoder"):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder
        self.news_embedding_dim = news_encoder.news_embedding_dim

        # TimeDistributed layer for processing history
        self.time_distributed = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_user"
        )

        # GraphSAGE layer
        self.graph_sage = GraphSAGELayer(
            units=self.news_embedding_dim,
            num_layers=config.graph_num_layers,
            aggregator='mean',
            dropout_rate=config.dropout_rate,
            seed=config.seed
        )

        # User node embedding (trainable parameter)
        # Note: In actual implementation, this should be dynamically sized based on batch
        self.user_node_embedding = self.add_weight(
            name='user_node_embedding',
            shape=(1, self.news_embedding_dim),
            initializer='zeros',
            trainable=True
        )

        # Attention layers
        self.K = layers.Dense(config.attention_dim, use_bias=False, name='attention_K')
        self.Q = layers.Dense(config.attention_dim, use_bias=True, name='attention_Q')

        # Other layers
        self.dropout = layers.Dropout(config.dropout_rate, seed=config.seed)

    def create_bipartite_graph(self, user_history_mask, batch_size):
        """Create adjacency matrix for user-news bipartite graph.
        
        Args:
            user_history_mask: (batch_size, max_history_num)
            batch_size: Batch size
            
        Returns:
            adjacency_matrix: (batch_size, max_history_num + 1, max_history_num + 1)
        """
        max_history_num = ops.shape(user_history_mask)[1]
        total_nodes = max_history_num + 1  # history nodes + user node

        # Create adjacency matrix
        adjacency = ops.zeros((batch_size, total_nodes, total_nodes))

        # Connect user node (last node) to all history nodes based on mask
        # User node is at index max_history_num
        user_node_idx = max_history_num

        # Create connections from user to history nodes
        user_history_mask_float = ops.cast(user_history_mask, dtype='float32')

        # Update adjacency matrix
        # adjacency[:, user_node_idx, :max_history_num] = user_history_mask_float
        # adjacency[:, :max_history_num, user_node_idx] = user_history_mask_float

        # For now, let's use a simpler approach with full connectivity
        # Perhaps it's a good idea to use sparse matrix handling
        adjacency = ops.ones((batch_size, total_nodes, total_nodes))

        return adjacency

    def call(self, inputs, training=None):
        """Forward pass for user encoding.
        
        Args:
            inputs: Tuple of (history_inputs, candidate_news_representation)
                - history_inputs: Concatenated history tensor (batch_size, history_length, feature_size)
                - candidate_news_representation: (batch_size, news_num, news_embedding_dim)
            training: Whether in training mode
            
        Returns:
            User representations (batch_size, news_num, news_embedding_dim)
        """
        history_inputs, candidate_news_representation = inputs

        batch_size = ops.shape(candidate_news_representation)[0]
        news_num = ops.shape(candidate_news_representation)[1]

        # Process all history items using TimeDistributed layer
        history_embedding = self.time_distributed(history_inputs, training=training)
        # Result: (batch_size, history_length, news_embedding_dim)

        # Create mask for valid history items by checking title tokens
        # Extract title tokens from the concatenated input to create the mask
        title_tokens = history_inputs[:, :, :self.config.max_title_length]
        history_mask = ops.any(ops.not_equal(title_tokens, 0), axis=-1)

        # Add user node embedding
        user_node_expanded = ops.repeat(self.user_node_embedding, batch_size, axis=0)
        user_node_expanded = ops.expand_dims(user_node_expanded, axis=1)

        # Concatenate history embeddings with user node
        all_embeddings = ops.concatenate([
            history_embedding,
            self.dropout(user_node_expanded, training=training)
        ], axis=1)

        # Create adjacency matrix for bipartite graph
        adjacency_matrix = self.create_bipartite_graph(history_mask, batch_size)

        # Apply GraphSAGE
        gcn_features = self.graph_sage((all_embeddings, adjacency_matrix), training=training)

        # Extract only history node features (exclude user node)
        max_history_num = ops.shape(history_embedding)[1]
        gcn_features = gcn_features[:, :max_history_num, :]

        # Expand for multiple candidates
        gcn_features_expanded = ops.expand_dims(gcn_features, axis=1)
        gcn_features_expanded = ops.repeat(gcn_features_expanded, news_num, axis=1)

        # Attention mechanism
        batch_news_num = batch_size * news_num

        # Reshape for attention computation
        gcn_features_reshaped = ops.reshape(
            gcn_features_expanded,
            (batch_news_num, max_history_num, self.news_embedding_dim)
        )
        candidate_reshaped = ops.reshape(
            candidate_news_representation,
            (batch_news_num, self.news_embedding_dim)
        )

        # Compute attention
        K = self.K(gcn_features_reshaped)
        Q = self.Q(candidate_reshaped)
        Q_expanded = ops.expand_dims(Q, axis=2)

        # Attention scores
        attention_scores = ops.matmul(K, Q_expanded)
        attention_scores = ops.squeeze(attention_scores, axis=-1)
        attention_scores = attention_scores / ops.sqrt(float(self.config.attention_dim))

        # Apply softmax
        attention_weights = ops.softmax(attention_scores, axis=1)
        attention_weights_expanded = ops.expand_dims(attention_weights, axis=1)

        # Weighted sum
        user_representation = ops.matmul(
            attention_weights_expanded,
            gcn_features_reshaped
        )
        user_representation = ops.squeeze(user_representation, axis=1)

        # Reshape back
        user_representation = ops.reshape(
            user_representation,
            (batch_size, news_num, self.news_embedding_dim)
        )

        return user_representation


class CROWNScorer(keras.Model):
    """Scoring component for CROWN model."""

    def __init__(self, config: CROWNConfig, news_encoder: CROWNNewsEncoder,
                 user_encoder: CROWNUserEncoder, name: str = "crown_scorer"):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder

        # Store TimeDistributed layers for candidate processing
        self.candidate_encoder_train = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_candidates"
        )
        self.candidate_encoder_eval = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_eval"
        )

    def score_training_batch(self, history_inputs, candidate_inputs, training=None):
        """Score training batch with softmax output.
        
        Args:
            history_inputs: Concatenated history tensor (batch_size, history_length, feature_size)
            candidate_inputs: Concatenated candidates tensor (batch_size, num_candidates, feature_size)
            training: Whether in training mode
            
        Returns:
            Softmax scores (batch_size, num_candidates)
        """
        # Get representations for all candidates using TimeDistributed layer
        candidate_repr = self.candidate_encoder_train(candidate_inputs, training=training)
        # Result: (batch_size, num_candidates, news_embedding_dim)

        # Encode user with candidates  
        user_repr = self.user_encoder((history_inputs, candidate_repr), training=training)

        # Calculate scores using element-wise multiplication and sum
        scores = ops.sum(user_repr * candidate_repr, axis=-1)

        # Apply softmax
        return ops.softmax(scores, axis=-1)

    def score_single_candidate(self, history_inputs, candidate_inputs, training=None):
        """Score single candidate with sigmoid output.
        
        Args:
            history_inputs: Concatenated history tensor (batch_size, history_length, feature_size)
            candidate_inputs: Concatenated candidate tensor (batch_size, feature_size)
            training: Whether in training mode
            
        Returns:
            Sigmoid scores (batch_size, 1)
        """
        # Encode candidate
        candidate_repr = self.news_encoder(candidate_inputs, training=training)
        candidate_repr = ops.expand_dims(candidate_repr, axis=1)

        # Encode user
        user_repr = self.user_encoder((history_inputs, candidate_repr), training=training)
        user_repr = ops.squeeze(user_repr, axis=1)

        # Calculate score
        score = ops.sum(user_repr * ops.squeeze(candidate_repr, axis=1), axis=-1)

        # Apply sigmoid
        return ops.sigmoid(score)

    def score_multiple_candidates(self, history_inputs, candidate_inputs, training=None):
        """Score multiple candidates with sigmoid scores.
        
        Args:
            history_inputs: Concatenated history tensor (batch_size, history_length, feature_size)
            candidate_inputs: Concatenated candidates tensor (batch_size, num_candidates, feature_size)
            training: Whether in training mode
            
        Returns:
            Sigmoid scores (batch_size, num_candidates)
        """
        # Get representations for all candidates using TimeDistributed layer
        candidate_repr = self.candidate_encoder_eval(candidate_inputs, training=training)
        # Result: (batch_size, num_candidates, news_embedding_dim)

        # Encode user
        user_repr = self.user_encoder((history_inputs, candidate_repr), training=training)

        # Calculate scores
        scores = ops.sum(user_repr * candidate_repr, axis=-1)

        # Apply sigmoid for consistency
        return ops.sigmoid(scores)


class CROWN(BaseModel):
    """CROWN: A Novel Approach to Comprehending Users' Preferences 
    for Accurate Personalized News Recommendation.
    
    This model implements:
    1. Category-guided intent disentanglement
    2. Consistency-based news representation
    3. GNN-enhanced hybrid user representation
    """

    def __init__(
            self,
            processed_news: Dict[str, Any],
            # Model dimensions
            embedding_size: int = 300,
            intent_embedding_dim: int = 200,
            category_embedding_dim: int = 100,
            subcategory_embedding_dim: int = 100,
            attention_dim: int = 200,
            # Intent disentanglement
            intent_num: int = 4,
            alpha: float = 0.5,
            # Transformer parameters
            num_heads: int = 16,
            head_dim: int = 16,
            feedforward_dim: int = 512,
            num_layers: int = 2,
            # ISAB parameters
            isab_num_heads: int = 4,
            isab_num_inducing_points: int = 4,
            # GraphSAGE parameters
            graph_hidden_dim: int = 300,
            graph_num_layers: int = 1,
            # Common parameters
            dropout_rate: float = 0.2,
            seed: int = 42,
            # Input parameters
            max_title_length: int = 50,
            max_abstract_length: int = 100,
            max_history_length: int = 50,
            max_impressions_length: int = 5,
            process_user_id: bool = False,
            name: str = "crown",
            **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Create configuration
        self.config = CROWNConfig(
            embedding_size=embedding_size,
            intent_embedding_dim=intent_embedding_dim,
            category_embedding_dim=category_embedding_dim,
            subcategory_embedding_dim=subcategory_embedding_dim,
            attention_dim=attention_dim,
            intent_num=intent_num,
            alpha=alpha,
            num_heads=num_heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            isab_num_heads=isab_num_heads,
            isab_num_inducing_points=isab_num_inducing_points,
            graph_hidden_dim=graph_hidden_dim,
            graph_num_layers=graph_num_layers,
            dropout_rate=dropout_rate,
            seed=seed,
            max_title_length=max_title_length,
            max_abstract_length=max_abstract_length,
            max_history_length=max_history_length,
            max_impressions_length=max_impressions_length,
            process_user_id=process_user_id,
        )

        # Store processed news data
        self.processed_news = processed_news
        self._validate_processed_news()

        # Initialize components
        self._create_components()

        # Set BaseModel attributes for fast evaluation
        self.process_user_id = process_user_id
        self.float_dtype = "float32"

    def _validate_processed_news(self):
        """Validate processed news data."""
        required_keys = ["vocab_size", "embeddings"]
        for key in required_keys:
            if key not in self.processed_news:
                raise ValueError(f"Missing required key '{key}' in processed_news")

        # Check for category information (could be either format)
        if "num_categories" in self.processed_news:
            self.processed_news["category_num"] = self.processed_news["num_categories"]
        if "num_subcategories" in self.processed_news:
            self.processed_news["subcategory_num"] = self.processed_news["num_subcategories"]

    def _create_components(self):
        """Create all model components."""
        # Create embedding layers
        self.embedding_layer = layers.Embedding(
            input_dim=self.processed_news["vocab_size"],
            output_dim=self.config.embedding_size,
            embeddings_initializer=keras.initializers.Constant(self.processed_news["embeddings"]),
            trainable=True,
            mask_zero=True,
            name="word_embedding",
        )

        self.category_embedding = layers.Embedding(
            input_dim=self.processed_news["category_num"],
            output_dim=self.config.category_embedding_dim,
            embeddings_initializer='glorot_uniform',
            trainable=True,
            name="category_embedding",
        )

        self.subcategory_embedding = layers.Embedding(
            input_dim=self.processed_news["subcategory_num"],
            output_dim=self.config.subcategory_embedding_dim,
            embeddings_initializer='glorot_uniform',
            trainable=True,
            name="subcategory_embedding",
        )

        # Create encoders
        self.news_encoder = CROWNNewsEncoder(
            self.config,
            self.embedding_layer,
            self.category_embedding,
            self.subcategory_embedding
        )

        # Set category predictor with correct number of categories
        category_num = self.processed_news.get("category_num", 18)  # Default from your data
        self.news_encoder.category_predictor = CategoryPredictor(category_num)

        self.user_encoder = CROWNUserEncoder(self.config, self.news_encoder)
        self.scorer = CROWNScorer(self.config, self.news_encoder, self.user_encoder)

        # Build compatibility models
        self.training_model, self.scorer_model = self._build_compatibility_models()

    def _build_compatibility_models(self) -> Tuple[keras.Model, keras.Model]:
        """Build training and scorer models for compatibility."""
        # ----- Training model -----
        # Create concatenated inputs that match the expected format
        history_concat_input = keras.Input(
            shape=(self.config.max_history_length, self.config.max_title_length + self.config.max_abstract_length + 2),
            dtype="int32", name="hist_concat_input"
        )
        candidate_concat_input = keras.Input(
            shape=(self.config.max_impressions_length,
                   self.config.max_title_length + self.config.max_abstract_length + 2),
            dtype="int32", name="cand_concat_input"
        )

        # Pass concatenated inputs directly to the scorer
        training_output = self.scorer.score_training_batch(history_concat_input, candidate_concat_input)

        training_model = keras.Model(
            inputs=[history_concat_input, candidate_concat_input],
            outputs=training_output,
            name="crown_training_model"
        )

        # ----- Scorer model -----
        # Create concatenated inputs for scorer model
        score_history_concat_input = keras.Input(
            shape=(self.config.max_history_length, self.config.max_title_length + self.config.max_abstract_length + 2),
            dtype="int32", name="score_hist_concat_input"
        )
        score_candidate_concat_input = keras.Input(
            shape=(self.config.max_title_length + self.config.max_abstract_length + 2,),
            dtype="int32", name="score_cand_concat_input"
        )

        # Pass concatenated inputs directly to the scorer
        scorer_output = self.scorer.score_single_candidate(score_history_concat_input, score_candidate_concat_input)

        scorer_model = keras.Model(
            inputs=[score_history_concat_input, score_candidate_concat_input],
            outputs=scorer_output,
            name="crown_scorer_model"
        )

        return training_model, scorer_model

    def call(self, inputs, training=None):
        """Main forward pass of CROWN model."""
        # Extract features based on input keys
        if training:
            return self._handle_training(inputs, training)
        else:
            # For inference, route based on input format
            if all(key in inputs for key in ['hist_tokens', 'cand_tokens']):
                return self._handle_multiple_candidates(inputs, training)
            else:
                raise ValueError("Invalid input format for CROWN model")

    def _handle_training(self, inputs, training=None):
        """Handle training batch scoring with softmax output."""
        # Concatenate history inputs
        history_concat = ops.concatenate([
            inputs["hist_tokens"],
            inputs["hist_abstract_tokens"],
            ops.expand_dims(inputs["hist_category"], axis=-1),
            ops.expand_dims(inputs["hist_subcategory"], axis=-1),
        ], axis=-1)

        # Concatenate candidate inputs
        candidate_concat = ops.concatenate([
            inputs["cand_tokens"],
            inputs["cand_abstract_tokens"],
            ops.expand_dims(inputs["cand_category"], axis=-1),
            ops.expand_dims(inputs["cand_subcategory"], axis=-1),
        ], axis=-1)

        return self.scorer.score_training_batch(history_concat, candidate_concat, training=training)

    def _handle_multiple_candidates(self, inputs, training=None):
        """Handle multiple candidate scoring with sigmoid scores."""
        # Concatenate history inputs
        history_concat = ops.concatenate([
            inputs["hist_tokens"],
            inputs["hist_abstract_tokens"],
            ops.expand_dims(inputs["hist_category"], axis=-1),
            ops.expand_dims(inputs["hist_subcategory"], axis=-1),
        ], axis=-1)

        # Concatenate candidate inputs
        candidate_concat = ops.concatenate([
            inputs["cand_tokens"],
            inputs["cand_abstract_tokens"],
            ops.expand_dims(inputs["cand_category"], axis=-1),
            ops.expand_dims(inputs["cand_subcategory"], axis=-1),
        ], axis=-1)

        return self.scorer.score_multiple_candidates(history_concat, candidate_concat, training=False)

    def get_config(self):
        """Return configuration for serialization."""
        base_config = super().get_config()
        base_config.update({
            "embedding_size": self.config.embedding_size,
            "intent_embedding_dim": self.config.intent_embedding_dim,
            "category_embedding_dim": self.config.category_embedding_dim,
            "subcategory_embedding_dim": self.config.subcategory_embedding_dim,
            "attention_dim": self.config.attention_dim,
            "intent_num": self.config.intent_num,
            "alpha": self.config.alpha,
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "feedforward_dim": self.config.feedforward_dim,
            "num_layers": self.config.num_layers,
            "isab_num_heads": self.config.isab_num_heads,
            "isab_num_inducing_points": self.config.isab_num_inducing_points,
            "graph_hidden_dim": self.config.graph_hidden_dim,
            "graph_num_layers": self.config.graph_num_layers,
            "dropout_rate": self.config.dropout_rate,
            "seed": self.config.seed,
            "max_title_length": self.config.max_title_length,
            "max_abstract_length": self.config.max_abstract_length,
            "max_history_length": self.config.max_history_length,
            "max_impressions_length": self.config.max_impressions_length,
            "process_user_id": self.config.process_user_id,
        })
        return base_config

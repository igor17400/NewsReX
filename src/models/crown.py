from typing import Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

import keras
from keras import layers, ops

from .layers import AdditiveAttentionLayer, GraphSAGELayer, InducedSetAttentionBlock, MultiHeadAttentionBlock
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
    intent_num: int = 3  # Number of intents (k)
    alpha: float = 0.3  # Weight for auxiliary loss

    # MAB parameters
    num_heads: int = 12  # 300 ÷ 12 = 25 (evenly divisible)
    head_dim: int = 25  # 12 × 25 = 300 = embedding_size
    feedforward_dim: int = 512
    num_layers: int = 2

    # ISAB parameters
    isab_num_heads: int = 12  # Match MAB heads for consistency
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
    """Positional encoding layer for MAB models."""

    def __init__(self, dropout_rate=0.1, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.dropout = layers.Dropout(dropout_rate)
        self.max_len = max_len
        self.supports_masking = True

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
        # Use broadcast_to to avoid dynamic slicing issues with JAX
        input_shape = ops.shape(x)
        batch_size, seq_len = input_shape[0], input_shape[1]
        
        # Select the appropriate slice of positional encoding and broadcast to match input
        pe_for_input = self.pe[:, :seq_len, :]
        pe_broadcasted = ops.broadcast_to(pe_for_input, (batch_size, seq_len, pe_for_input.shape[-1]))
        
        x = x + pe_broadcasted
        return self.dropout(x, training=training)

    def compute_mask(self, inputs, mask=None):
        """Pass through the mask unchanged."""
        return mask


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


class NewsEncoder(keras.Model):
    """News encoder component for CROWN model.

    Implements category-guided intent disentanglement and consistency-based
    news representation.
    """

    def __init__(self, config: CROWNConfig, embedding_layer: layers.Embedding,
                 category_embedding: layers.Embedding, subcategory_embedding: layers.Embedding,
                 name: str = "news_encoder"):
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

        # MAB encoders
        self.title_mab = self._build_mab_encoder("title")
        self.body_mab = self._build_mab_encoder("body")

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

        # Category predictor for auxiliary task - initialized with default, will be overridden by parent
        self.category_predictor = CategoryPredictor(18)  # Default, will be replaced in _create_components

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the news encoder.

        Args:
            input_shape: Input shape (batch_size, title_length + abstract_length + 2)

        Returns:
            Output shape (batch_size, news_embedding_dim)
        """
        return (input_shape[0], self.news_embedding_dim)

    def _build_mab_encoder(self, name_prefix):
        """Build an encoder using single MultiHeadAttentionBlock from layers.py.

        According to the CROWN paper, we use a single MAB (Multi-Head Attention Block):
        MAB(X,Y) = LayerNorm(H + FeedForward(H)) where H = LayerNorm(X + Multihead(X,Y,Y))
        """

        # Create single MAB block as per CROWN paper
        mab_block = MultiHeadAttentionBlock(
            dim_out=self.config.embedding_size,
            num_heads=self.config.num_heads,
            use_layer_norm=True,
            seed=self.config.seed,
            name=f'{name_prefix}_mab_block'
        )

        return mab_block

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
        # Use fixed dimensions to avoid dynamic conditions that break JAX tracing
        title_tokens = inputs[:, :self.config.max_title_length]
        abstract_tokens = inputs[
            :, self.config.max_title_length:self.config.max_title_length + self.config.max_abstract_length]
        category_id = inputs[
            :, self.config.max_title_length + self.config.max_abstract_length:self.config.max_title_length + self.config.max_abstract_length + 1]
        subcategory_id = inputs[:, self.config.max_title_length + self.config.max_abstract_length + 1:]

        # Squeeze dimension
        category_id = ops.squeeze(category_id, axis=-1)
        subcategory_id = ops.squeeze(subcategory_id, axis=-1)

        # Word embeddings
        title_w = self.dropout(self.embedding_layer(title_tokens), training=training)
        body_w = self.dropout(self.embedding_layer(abstract_tokens), training=training)

        # Positional encoding
        title_p = self.title_pos_encoder(title_w, training=training)
        body_p = self.body_pos_encoder(body_w, training=training)

        # MAB encoding
        title_t = self.title_mab(title_p, training=training)
        body_t = self.body_mab(body_p, training=training)

        # Mean pooling
        title_embedding = ops.mean(title_t, axis=1)
        body_embedding = ops.mean(body_t, axis=1)

        # Category-aware intent disentanglement
        cat_embed = self.category_embedding(category_id)
        subcat_embed = self.subcategory_embedding(subcategory_id)
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
        # Always compute the auxiliary loss - avoiding conditional logic for JAX tracing
        category_logits = self.category_predictor(title_intent_embedding)
        category_one_hot = ops.one_hot(category_id, self.category_predictor.category_num)
        category_loss = ops.categorical_crossentropy(
            category_one_hot,
            category_logits,
            from_logits=True
        )
        auxiliary_loss = ops.mean(category_loss) * self.config.alpha

        # Always add the loss - Keras will handle it appropriately based on training mode
        self.add_loss(auxiliary_loss)

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


class UserEncoder(keras.Model):
    """User encoder component for CROWN model.

    Implements GNN-enhanced hybrid user representation using GraphSAGE.
    """

    def __init__(self, config: CROWNConfig, news_encoder: NewsEncoder,
                 name: str = "user_encoder"):
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

        # User node embeddings - one per batch item
        # Note: The reference uses batch_size in the parameter, but we'll make it dynamic
        self.user_node_embedding_init = self.add_weight(
            name='user_node_embedding_init',
            shape=(self.news_embedding_dim,),
            initializer='zeros',
            trainable=True
        )

        # Attention layers
        self.K = layers.Dense(config.attention_dim, use_bias=False, name='attention_K')
        self.Q = layers.Dense(config.attention_dim, use_bias=True, name='attention_Q')

        # Other layers
        self.dropout = layers.Dropout(config.dropout_rate, seed=config.seed)

    def build(self, input_shape):
        super().build(input_shape)

    def create_bipartite_graph(self, user_history_mask, batch_size):
        """Create adjacency matrix for user-news bipartite graph.

        Creates a simpler bipartite graph where each user (single node per batch)
        connects to their valid history nodes. This is more memory efficient.

        Args:
            user_history_mask: (batch_size, max_history_num)
            batch_size: Batch size

        Returns:
            adjacency_matrix: (batch_size, max_history_num + 1, max_history_num + 1)
        """
        # Use fixed max_history_length from config to avoid dynamic shapes
        max_history_num = self.config.max_history_length

        # Convert mask to float
        user_history_mask_float = ops.cast(user_history_mask, dtype='float32')

        # Create adjacency matrix parts
        # Top-left: history-to-history (no connections)
        hist_to_hist = ops.zeros((batch_size, max_history_num, max_history_num))

        # Top-right: history-to-user connections based on mask
        hist_to_user = ops.expand_dims(user_history_mask_float, axis=-1)

        # Bottom-left: user-to-history connections based on mask
        user_to_hist = ops.expand_dims(user_history_mask_float, axis=1)

        # Bottom-right: user-to-user (no self-loop)
        user_to_user = ops.zeros((batch_size, 1, 1))

        # Combine all parts
        top_half = ops.concatenate([hist_to_hist, hist_to_user], axis=2)
        bottom_half = ops.concatenate([user_to_hist, user_to_user], axis=2)
        adjacency = ops.concatenate([top_half, bottom_half], axis=1)

        return adjacency

    def call(self, inputs, training=None):
        """Forward pass for user encoding.

        Args:
            inputs: Concatenated history tensor (batch_size, history_length, feature_size)
            training: Whether in training mode

        Returns:
            User representations (batch_size, news_embedding_dim)
        """
        history_inputs = inputs
        # Avoid dynamic batch size - use a static approach
        
        # Process all history items using TimeDistributed layer
        history_embedding = self.time_distributed(history_inputs, training=training)
        # Result: (batch_size, history_length, news_embedding_dim)

        # Create mask for valid history items by checking title tokens
        # Extract title tokens from the concatenated input to create the mask
        title_tokens = history_inputs[:, :, :self.config.max_title_length]
        history_mask = ops.any(ops.not_equal(title_tokens, 0), axis=-1)

        # Create user node embeddings - use broadcasting instead of repeat with dynamic batch size
        batch_size = ops.shape(history_inputs)[0]  # Only use this where necessary
        user_node_embeddings = ops.expand_dims(self.user_node_embedding_init, axis=0)
        user_node_embeddings = ops.broadcast_to(user_node_embeddings, (batch_size, self.news_embedding_dim))
        user_node_embeddings = ops.expand_dims(user_node_embeddings, axis=1)

        # Concatenate history embeddings with user node embeddings
        # Shape: [batch_size, max_history_num + 1, news_embedding_dim]
        all_embeddings = ops.concatenate([
            history_embedding,
            self.dropout(user_node_embeddings, training=training)
        ], axis=1)

        # Create adjacency matrix for bipartite graph
        adjacency_matrix = self.create_bipartite_graph(history_mask, batch_size)

        # Apply GraphSAGE
        gcn_features = self.graph_sage((all_embeddings, adjacency_matrix), training=training)

        # Extract only history node features (exclude user node)
        # Use fixed max_history_length from config to avoid dynamic shapes
        max_history_num = self.config.max_history_length
        gcn_features = gcn_features[:, :max_history_num, :]

        # Apply mask and mean pooling over history to get user representation
        mask_expanded = ops.expand_dims(ops.cast(history_mask, dtype='float32'), axis=-1)
        masked_gcn = gcn_features * mask_expanded

        # Compute mean, avoiding division by zero
        sum_features = ops.sum(masked_gcn, axis=1)
        count_features = ops.sum(mask_expanded, axis=1)
        count_features = ops.maximum(count_features, 1.0)  # Avoid division by zero
        user_representation = sum_features / count_features

        return user_representation


class CROWNScorer(keras.Model):
    """Scoring component for CROWN model."""

    def __init__(self, config: CROWNConfig, news_encoder: NewsEncoder,
                 user_encoder: UserEncoder, name: str = "crown_scorer"):
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
        # Always use training=True for training batch
        candidate_repr = self.candidate_encoder_train(candidate_inputs, training=True)
        # Result: (batch_size, num_candidates, news_embedding_dim)

        # Encode user
        user_repr = self.user_encoder(history_inputs, training=True)

        # Expand user representation for multiple candidates
        user_repr_expanded = ops.expand_dims(user_repr, axis=1)
        user_repr_expanded = ops.repeat(user_repr_expanded, self.config.max_impressions_length, axis=1)

        # Calculate scores using element-wise multiplication and sum
        scores = ops.sum(user_repr_expanded * candidate_repr, axis=-1)

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
        candidate_repr = self.news_encoder(candidate_inputs, training=False)
        candidate_repr = ops.expand_dims(candidate_repr, axis=1)

        # Encode user
        user_repr = self.user_encoder(history_inputs, training=False)

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
        candidate_repr = self.candidate_encoder_eval(candidate_inputs, training=False)
        # Result: (batch_size, num_candidates, news_embedding_dim)

        # Encode user
        user_repr = self.user_encoder(history_inputs, training=False)

        # Expand user representation for multiple candidates
        user_repr_expanded = ops.expand_dims(user_repr, axis=1)
        user_repr_expanded = ops.repeat(user_repr_expanded, self.config.max_impressions_length, axis=1)

        # Calculate scores
        scores = ops.sum(user_repr_expanded * candidate_repr, axis=-1)

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
            # MAB parameters
            num_heads: int = 12,
            head_dim: int = 25,
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
            mask_zero=False,
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
        self.news_encoder = NewsEncoder(
            self.config,
            self.embedding_layer,
            self.category_embedding,
            self.subcategory_embedding
        )

        # Set category predictor with correct number of categories
        category_num = self.processed_news.get("category_num", 18)  # Default from your data
        self.news_encoder.category_predictor = CategoryPredictor(category_num)

        self.user_encoder = UserEncoder(self.config, self.news_encoder)
        self.scorer = CROWNScorer(self.config, self.news_encoder, self.user_encoder)

        # Build compatibility models
        self.training_model, self.scorer_model = self._build_compatibility_models()

    def build(self, input_shape=None):
        """Build the CROWN model with proper layer initialization.
        
        This method ensures all internal layers are properly built
        when the model is first called.
        """
        if self.built:
            return
            
        # Build all components to ensure proper initialization
        # Note: The components are already created in __init__, this just marks them as built
        if hasattr(self, 'embedding_layer'):
            self.embedding_layer.build((None,))  # Build with variable batch size
        
        if hasattr(self, 'news_encoder') and hasattr(self.news_encoder, 'build'):
            self.news_encoder.build((None, self.config.max_title_length + self.config.max_abstract_length + 2))
            
        if hasattr(self, 'user_encoder') and hasattr(self.user_encoder, 'build'):
            self.user_encoder.build((None, self.config.max_history_length, 
                                   self.config.max_title_length + self.config.max_abstract_length + 2))
            
        if hasattr(self, 'scorer') and hasattr(self.scorer, 'build'):
            self.scorer.build(None)
        
        super().build(input_shape)

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

    def _validate_inputs(self, inputs: Dict, training: bool = None) -> None:
        """Validate input format and shapes based on mode."""
        if not isinstance(inputs, dict):
            raise TypeError("Inputs must be a dictionary")

        input_keys = set(inputs.keys())

        if training:
            # Training mode expects multiple keys for history and candidates
            required_hist_keys = {"hist_tokens", "hist_abstract_tokens", "hist_category", "hist_subcategory"}
            required_cand_keys = {"cand_tokens", "cand_abstract_tokens", "cand_category", "cand_subcategory"}

            if not (required_hist_keys.issubset(input_keys) and required_cand_keys.issubset(input_keys)):
                raise ValueError(
                    f"Training mode requires history keys: {required_hist_keys} and candidate keys: {required_cand_keys}, "
                    f"but got keys: {list(input_keys)}"
                )
        else:
            # Inference mode - check for different valid combinations
            # Add validation logic for inference mode if needed
            pass

    def call(self, inputs, training=None):
        """Main forward pass of CROWN model."""
        self._validate_inputs(inputs, training)

        if training:
            # Training mode: always use training format
            return self._handle_training(inputs)
        else:
            # Inference mode: route based on input format
            if "hist_tokens" in inputs and "cand_tokens" in inputs:
                # Multiple candidates format
                return self._handle_multiple_candidates(inputs)
            else:
                # Add other inference modes as needed
                raise ValueError("Invalid input format for inference mode")

    def _handle_training(self, inputs):
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

        return self.scorer.score_training_batch(history_concat, candidate_concat)

    def _handle_multiple_candidates(self, inputs):
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

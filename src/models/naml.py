from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import keras
from keras import layers, ops

from .layers import AdditiveAttentionLayer
from .base import BaseModel


@dataclass
class NAMLConfig:
    """Configuration class for NAML model parameters."""
    max_title_length: int = 30
    max_abstract_length: int = 50
    embedding_size: int = 300
    category_embedding_dim: int = 100
    subcategory_embedding_dim: int = 100
    cnn_filter_num: int = 400
    cnn_kernel_size: int = 3
    word_attention_query_dim: int = 200
    view_attention_query_dim: int = 200
    user_attention_query_dim: int = 200
    dropout_rate: float = 0.2
    activation: str = "relu"
    max_history_length: int = 50
    max_impressions_length: int = 5
    process_user_id: bool = False
    seed: int = 42


class TitleEncoder(keras.Model):
    """Title encoder component for NAML.
    
    Processes news titles through word embeddings, CNN, and additive attention.
    """

    def __init__(self, config: NAMLConfig, embedding_layer: layers.Embedding, name: str = "title_encoder"):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer

        # Create layers
        self.dropout1 = layers.Dropout(self.config.dropout_rate, seed=self.config.seed, name="title_embedding_dropout")
        self.cnn = layers.Conv1D(
            self.config.cnn_filter_num,
            self.config.cnn_kernel_size,
            activation=self.config.activation,
            padding="same",
            name="title_cnn",
        )
        self.dropout2 = layers.Dropout(self.config.dropout_rate, seed=self.config.seed, name="title_cnn_dropout")
        self.additive_attention = AdditiveAttentionLayer(
            self.config.word_attention_query_dim,
            seed=self.config.seed,
            name="title_word_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the title encoder.
        
        Args:
            input_shape: Input shape (batch_size, title_length)
            
        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape[0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for title encoding.
        
        Args:
            inputs: Title token sequences (batch_size, title_length)
            training: Whether in training mode
            
        Returns:
            Title representations (batch_size, cnn_filter_num)
        """
        # Word Embedding
        embedded_sequences = self.embedding_layer(inputs)

        # Dropout after embedding
        y = self.dropout1(embedded_sequences, training=training)

        # CNN
        y = self.cnn(y)

        # Dropout after CNN
        y = self.dropout2(y, training=training)

        # Create padding mask for attention
        padding_mask = ops.not_equal(inputs, 0)

        # Additive Attention to get single title vector
        title_representation = self.additive_attention(y, mask=padding_mask)

        return title_representation


class AbstractEncoder(keras.Model):
    """Abstract encoder component for NAML.
    
    Processes news abstracts through word embeddings, CNN, and additive attention.
    """

    def __init__(self, config: NAMLConfig, embedding_layer: layers.Embedding, name: str = "abstract_encoder"):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer

        # Create layers
        self.dropout1 = layers.Dropout(self.config.dropout_rate, seed=self.config.seed,
                                       name="abstract_embedding_dropout")
        self.cnn = layers.Conv1D(
            self.config.cnn_filter_num,
            self.config.cnn_kernel_size,
            activation=self.config.activation,
            padding="same",
            name="abstract_cnn",
        )
        self.dropout2 = layers.Dropout(self.config.dropout_rate, seed=self.config.seed, name="abstract_cnn_dropout")
        self.additive_attention = AdditiveAttentionLayer(
            self.config.word_attention_query_dim,
            seed=self.config.seed,
            name="abstract_word_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the abstract encoder.
        
        Args:
            input_shape: Input shape (batch_size, abstract_length)
            
        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape[0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for abstract encoding.
        
        Args:
            inputs: Abstract token sequences (batch_size, abstract_length)
            training: Whether in training mode
            
        Returns:
            Abstract representations (batch_size, cnn_filter_num)
        """
        # Word Embedding
        embedded_sequences = self.embedding_layer(inputs)

        # Dropout after embedding
        y = self.dropout1(embedded_sequences, training=training)

        # CNN
        y = self.cnn(y)

        # Dropout after CNN
        y = self.dropout2(y, training=training)

        # Create padding mask for attention
        padding_mask = ops.not_equal(inputs, 0)

        # Additive Attention to get single abstract vector
        abstract_representation = self.additive_attention(y, mask=padding_mask)

        return abstract_representation


class CategoryEncoder(keras.Model):
    """Category encoder component for NAML.
    
    Embeds and projects category IDs into the same dimension as CNN outputs.
    """

    def __init__(self, config: NAMLConfig, num_categories: int, name: str = "category_encoder"):
        super().__init__(name=name)
        self.config = config
        self.num_categories = num_categories

        # Create layers
        self.embedding = layers.Embedding(
            self.num_categories + 1,
            self.config.category_embedding_dim,
            trainable=True,
            name="category_embedding",
        )
        self.projection = layers.Dense(
            self.config.cnn_filter_num,
            activation=self.config.activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.config.seed),
            name="category_projection",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the category encoder.
        
        Args:
            input_shape: Input shape (batch_size, 1)
            
        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape[0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for category encoding.
        
        Args:
            inputs: Category IDs (batch_size, 1)
            training: Whether in training mode
            
        Returns:
            Category representations (batch_size, cnn_filter_num)
        """
        # Embedding
        embedded = self.embedding(inputs)

        # Projection to cnn_filter_num dimensions
        projected = self.projection(embedded)

        # Reshape from (batch_size, 1, cnn_filter_num) to (batch_size, cnn_filter_num)
        category_representation = ops.squeeze(projected, axis=1)

        return category_representation


class SubcategoryEncoder(keras.Model):
    """Subcategory encoder component for NAML.
    
    Embeds and projects subcategory IDs into the same dimension as CNN outputs.
    """

    def __init__(self, config: NAMLConfig, num_subcategories: int, name: str = "subcategory_encoder"):
        super().__init__(name=name)
        self.config = config
        self.num_subcategories = num_subcategories

        # Create layers
        self.embedding = layers.Embedding(
            self.num_subcategories + 1,
            self.config.subcategory_embedding_dim,
            trainable=True,
            name="subcategory_embedding",
        )
        self.projection = layers.Dense(
            self.config.cnn_filter_num,
            activation=self.config.activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.config.seed),
            name="subcategory_projection",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the subcategory encoder.
        
        Args:
            input_shape: Input shape (batch_size, 1)
            
        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape[0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for subcategory encoding.
        
        Args:
            inputs: Subcategory IDs (batch_size, 1)
            training: Whether in training mode
            
        Returns:
            Subcategory representations (batch_size, cnn_filter_num)
        """
        # Embedding
        embedded = self.embedding(inputs)

        # Projection to cnn_filter_num dimensions
        projected = self.projection(embedded)

        # Reshape from (batch_size, 1, cnn_filter_num) to (batch_size, cnn_filter_num)
        subcategory_representation = ops.squeeze(projected, axis=1)

        return subcategory_representation


class NewsEncoder(keras.Model):
    """News encoder component for NAML.
    
    Combines multiple views (title, abstract, category, subcategory) of a news article
    using view-level attention to produce a unified news representation.
    """

    def __init__(self,
                 config: NAMLConfig,
                 title_encoder: TitleEncoder,
                 abstract_encoder: AbstractEncoder,
                 category_encoder: CategoryEncoder,
                 subcategory_encoder: SubcategoryEncoder,
                 name: str = "news_encoder"):
        super().__init__(name=name)
        self.config = config
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.category_encoder = category_encoder
        self.subcategory_encoder = subcategory_encoder

        # View-level attention
        self.view_attention = AdditiveAttentionLayer(
            self.config.view_attention_query_dim,
            seed=self.config.seed,
            name="view_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the news encoder.
        
        Args:
            input_shape: Input shape dict with keys: title_tokens, abstract_tokens, category_id, subcategory_id
            
        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape["title_tokens"][0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for news encoding.
        
        Args:
            inputs: Concatenated tensor (batch_size, title_length + abstract_length + 2)
                   containing [title_tokens, abstract_tokens, category_id, subcategory_id]
            training: Whether in training mode
            
        Returns:
            News representations (batch_size, cnn_filter_num)
        """
        # Split the concatenated input: [title, abstract, category, subcategory]
        title_tokens = inputs[:, :self.config.max_title_length]
        abstract_tokens = inputs[:, self.config.max_title_length:self.config.max_title_length + self.config.max_abstract_length]
        category_id = inputs[:, self.config.max_title_length + self.config.max_abstract_length:self.config.max_title_length + self.config.max_abstract_length + 1]
        subcategory_id = inputs[:, self.config.max_title_length + self.config.max_abstract_length + 1:]

        # Encode each view
        title_vec = self.title_encoder(title_tokens, training=training)
        abstract_vec = self.abstract_encoder(abstract_tokens, training=training)
        category_vec = self.category_encoder(category_id, training=training)
        subcategory_vec = self.subcategory_encoder(subcategory_id, training=training)

        # Stack views for attention (batch_size, 4, cnn_filter_num)
        views = ops.stack([title_vec, abstract_vec, category_vec, subcategory_vec], axis=1)

        # Apply view-level attention to combine views
        news_representation = self.view_attention(views)

        return news_representation


class UserEncoder(keras.Model):
    """User encoder component for NAML.
    
    Processes user browsing history through news encoder and additive attention
    to produce user representations.
    """

    def __init__(self, config: NAMLConfig, news_encoder: NewsEncoder, name: str = "user_encoder"):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder

        # User-level attention
        self.user_attention = AdditiveAttentionLayer(
            self.config.user_attention_query_dim,
            seed=self.config.seed,
            name="user_additive_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the user encoder.
        
        Args:
            input_shape: Input shape dict with history of news items
            
        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape["title_tokens"][0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for user encoding.
        
        Args:
            inputs: Concatenated tensor (batch_size, history_length, title_length + abstract_length + 2)
                   containing user's browsing history
            training: Whether in training mode
            
        Returns:
            User representations (batch_size, cnn_filter_num)
        """
        batch_size = ops.shape(inputs)[0]
        history_length = ops.shape(inputs)[1]
        
        # Split the concatenated input: [title, abstract, category, subcategory]
        title_tokens = inputs[:, :, :self.config.max_title_length]
        abstract_tokens = inputs[:, :, self.config.max_title_length:self.config.max_title_length + self.config.max_abstract_length]
        category_id = inputs[:, :, self.config.max_title_length + self.config.max_abstract_length:self.config.max_title_length + self.config.max_abstract_length + 1]
        subcategory_id = inputs[:, :, self.config.max_title_length + self.config.max_abstract_length + 1:]

        # Reshape inputs to process all history items at once
        title_tokens_flat = ops.reshape(title_tokens, (-1, self.config.max_title_length))
        abstract_tokens_flat = ops.reshape(abstract_tokens, (-1, self.config.max_abstract_length))
        category_id_flat = ops.reshape(category_id, (-1, 1))
        subcategory_id_flat = ops.reshape(subcategory_id, (-1, 1))

        # Process all history items through news encoder using concatenated format
        news_input_flat = ops.concatenate([
            title_tokens_flat,
            abstract_tokens_flat, 
            category_id_flat,
            subcategory_id_flat
        ], axis=-1)
        news_vectors_flat = self.news_encoder(news_input_flat, training=training)

        # Reshape back to (batch_size, history_length, cnn_filter_num)
        news_vectors = ops.reshape(news_vectors_flat, (batch_size, history_length, self.config.cnn_filter_num))

        # Create mask for valid history items
        history_mask = ops.any(ops.not_equal(title_tokens, 0), axis=-1)

        # Apply user-level attention
        user_representation = self.user_attention(news_vectors, mask=history_mask)

        return user_representation


class NAMLScorer(keras.Model):
    """Scoring component for NAML.
    
    Handles different scoring scenarios: training (multiple candidates with softmax),
    single candidate scoring (with sigmoid), and multiple candidate scoring (raw scores).
    """

    def __init__(self, config: NAMLConfig, news_encoder: NewsEncoder, user_encoder: UserEncoder,
                 name: str = "naml_scorer"):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder

    def build(self, input_shape):
        super().build(input_shape)

    def score_training_batch(self, history_inputs, candidate_inputs, training=None):
        """Score training batch with softmax output."""
        # Get user representation
        user_repr = self.user_encoder(history_inputs, training=training)

        # Process candidates using TimeDistributed-like approach
        # Instead of reshaping, we'll process each candidate position separately
        batch_size = ops.shape(candidate_inputs["title_tokens"])[0]
        num_candidates = ops.shape(candidate_inputs["title_tokens"])[1]

        candidate_scores = []
        for i in range(self.config.max_impressions_length):  # Use fixed loop over max_impressions_length
            # Extract single candidate for all batches
            single_cand_inputs = {
                "title_tokens": candidate_inputs["title_tokens"][:, i, :],
                "abstract_tokens": candidate_inputs["abstract_tokens"][:, i, :],
                "category_id": candidate_inputs["category_id"][:, i, :],
                "subcategory_id": candidate_inputs["subcategory_id"][:, i, :],
            }

            # Get representation for this candidate
            cand_repr = self.news_encoder(single_cand_inputs, training=training)

            # Compute score with user representation
            score = ops.sum(cand_repr * user_repr, axis=-1, keepdims=True)
            candidate_scores.append(score)

        # Stack scores
        scores = ops.concatenate(candidate_scores, axis=-1)

        # Apply softmax for training
        return ops.softmax(scores, axis=-1)

    def score_single_candidate(self, history_inputs, candidate_inputs, training=None):
        """Score single candidate with sigmoid output."""
        # Get user representation
        user_repr = self.user_encoder(history_inputs, training=training)

        # Get candidate representation
        candidate_repr = self.news_encoder(candidate_inputs, training=training)

        # Calculate score using dot product
        score = ops.sum(candidate_repr * user_repr, axis=-1, keepdims=True)

        # Apply sigmoid for probability
        return ops.sigmoid(score)

    def score_multiple_candidates(self, history_inputs, candidate_inputs, training=False):
        """Score multiple candidates with sigmoid activation.
        
        This method efficiently processes all candidates using vectorized operations
        and applies sigmoid activation to maintain consistency with single candidate scoring.
        """
        # Get user representation
        user_repr = self.user_encoder(history_inputs, training=training)

        # Process candidates
        batch_size = ops.shape(candidate_inputs["title_tokens"])[0]
        num_candidates = ops.shape(candidate_inputs["title_tokens"])[1]

        # Reshape candidates to process all at once
        cand_inputs_flat = {
            "title_tokens": ops.reshape(candidate_inputs["title_tokens"], (-1, self.config.max_title_length)),
            "abstract_tokens": ops.reshape(candidate_inputs["abstract_tokens"], (-1, self.config.max_abstract_length)),
            "category_id": ops.reshape(candidate_inputs["category_id"], (-1, 1)),
            "subcategory_id": ops.reshape(candidate_inputs["subcategory_id"], (-1, 1)),
        }

        # Get candidate representations
        candidate_repr_flat = self.news_encoder(cand_inputs_flat, training=training)
        candidate_repr = ops.reshape(candidate_repr_flat, (batch_size, num_candidates, self.config.cnn_filter_num))

        # Calculate scores using dot product
        user_repr_expanded = ops.expand_dims(user_repr, axis=1)
        scores = ops.sum(candidate_repr * user_repr_expanded, axis=-1)

        # Apply sigmoid activation for consistency with single candidate scoring
        return ops.sigmoid(scores)


class NAML(BaseModel):
    """Neural Attentive Multi-View Learning (NAML) model for news recommendation.

    This model is based on the paper: "Neural News Recommendation with Attentive Multi-View Learning"
    by C. Wu et al. It learns news representations from multiple views (title, abstract, category)
    and user representations from their browsing history.

    Key features:
    - Multi-view news encoding: Processes title, abstract, and categories separately.
    - View-level attention: Combines different news views into a unified representation.
    - Attentive user encoding: Uses additive attention over historical news.
    
    Refactored with clean architecture using separate components for better
    maintainability, testability, and code organization.
    """

    def __init__(
            self,
            processed_news: Dict[str, Any],
            max_title_length: int = 30,
            max_abstract_length: int = 50,
            embedding_size: int = 300,
            category_embedding_dim: int = 100,
            subcategory_embedding_dim: int = 100,
            cnn_filter_num: int = 400,
            cnn_kernel_size: int = 3,
            word_attention_query_dim: int = 200,
            view_attention_query_dim: int = 200,
            user_attention_query_dim: int = 200,
            dropout_rate: float = 0.2,
            activation: str = "relu",
            max_history_length: int = 50,
            max_impressions_length: int = 5,
            process_user_id: bool = False,
            seed: int = 42,
            name: str = "naml",
            **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Create configuration object
        self.config = NAMLConfig(
            max_title_length=max_title_length,
            max_abstract_length=max_abstract_length,
            embedding_size=embedding_size,
            category_embedding_dim=category_embedding_dim,
            subcategory_embedding_dim=subcategory_embedding_dim,
            cnn_filter_num=cnn_filter_num,
            cnn_kernel_size=cnn_kernel_size,
            word_attention_query_dim=word_attention_query_dim,
            view_attention_query_dim=view_attention_query_dim,
            user_attention_query_dim=user_attention_query_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            max_history_length=max_history_length,
            max_impressions_length=max_impressions_length,
            process_user_id=process_user_id,
            seed=seed,
        )

        # Store processed news data
        self.processed_news = processed_news
        self._validate_processed_news()

        # Initialize components
        self._create_components()

        # Set BaseModel attributes for fast evaluation (required by BaseModel)
        self.process_user_id = process_user_id
        self.float_dtype = "float32"

    def _validate_processed_news(self) -> None:
        """Validate processed news data integrity."""
        required_keys = ["vocab_size", "embeddings", "num_categories", "num_subcategories"]
        for key in required_keys:
            if key not in self.processed_news:
                raise ValueError(f"Missing required key '{key}' in processed_news")

        embeddings_matrix = self.processed_news["embeddings"]
        if np.isnan(embeddings_matrix).any():
            raise ValueError("Embeddings matrix contains NaN values")

        if embeddings_matrix.shape[1] != self.config.embedding_size:
            raise ValueError(
                f"Embeddings dimension {embeddings_matrix.shape[1]} doesn't match "
                f"configured embedding_size {self.config.embedding_size}"
            )

    def _create_components(self) -> None:
        """Create all model components in proper order."""
        # Set random seed
        keras.utils.set_random_seed(self.config.seed)

        # Create shared embedding layer
        self.embedding_layer = layers.Embedding(
            input_dim=self.processed_news["vocab_size"],
            output_dim=self.config.embedding_size,
            embeddings_initializer=keras.initializers.Constant(self.processed_news["embeddings"]),
            trainable=True,
            mask_zero=False,
            name="word_embedding",
        )

        # Create view encoders
        self.title_encoder = TitleEncoder(self.config, self.embedding_layer)
        self.abstract_encoder = AbstractEncoder(self.config, self.embedding_layer)
        self.category_encoder = CategoryEncoder(
            self.config, self.processed_news["num_categories"]
        )
        self.subcategory_encoder = SubcategoryEncoder(
            self.config, self.processed_news["num_subcategories"]
        )

        # Create news encoder that combines all views
        self.news_encoder = NewsEncoder(
            self.config,
            self.title_encoder,
            self.abstract_encoder,
            self.category_encoder,
            self.subcategory_encoder,
        )

        # Create user encoder
        self.user_encoder = UserEncoder(self.config, self.news_encoder)

        # Create scorer
        self.scorer = NAMLScorer(self.config, self.news_encoder, self.user_encoder)

        # Build training and scorer models for compatibility
        self.training_model, self.scorer_model = self._build_compatibility_models()

    def _build_compatibility_models(self) -> Tuple[keras.Model, keras.Model]:
        """Build training and scorer models for backward compatibility."""
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

        # Split the concatenated inputs
        hist_title = history_concat_input[:, :, :self.config.max_title_length]
        hist_abstract = history_concat_input[
            :, :, self.config.max_title_length:self.config.max_title_length + self.config.max_abstract_length]
        hist_category = history_concat_input[
            :, :, self.config.max_title_length + self.config.max_abstract_length:self.config.max_title_length + self.config.max_abstract_length + 1]
        hist_subcategory = history_concat_input[
            :, :, self.config.max_title_length + self.config.max_abstract_length + 1:]

        cand_title = candidate_concat_input[:, :, :self.config.max_title_length]
        cand_abstract = candidate_concat_input[
            :, :, self.config.max_title_length:self.config.max_title_length + self.config.max_abstract_length]
        cand_category = candidate_concat_input[
            :, :, self.config.max_title_length + self.config.max_abstract_length:self.config.max_title_length + self.config.max_abstract_length + 1]
        cand_subcategory = candidate_concat_input[
            :, :, self.config.max_title_length + self.config.max_abstract_length + 1:]

        history_inputs = {
            "title_tokens": hist_title,
            "abstract_tokens": hist_abstract,
            "category_id": hist_category,
            "subcategory_id": hist_subcategory,
        }

        candidate_inputs = {
            "title_tokens": cand_title,
            "abstract_tokens": cand_abstract,
            "category_id": cand_category,
            "subcategory_id": cand_subcategory,
        }

        training_output = self.scorer.score_training_batch(history_inputs, candidate_inputs)

        training_model = keras.Model(
            inputs=[history_concat_input, candidate_concat_input],
            outputs=training_output,
            name="naml_training_model"
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

        # Split history input
        score_hist_title = score_history_concat_input[:, :, :self.config.max_title_length]
        score_hist_abstract = score_history_concat_input[
            :, :, self.config.max_title_length:self.config.max_title_length + self.config.max_abstract_length]
        score_hist_category = score_history_concat_input[
            :, :, self.config.max_title_length + self.config.max_abstract_length:self.config.max_title_length + self.config.max_abstract_length + 1]
        score_hist_subcategory = score_history_concat_input[
            :, :, self.config.max_title_length + self.config.max_abstract_length + 1:]

        # Split candidate input
        score_cand_title = score_candidate_concat_input[:, :self.config.max_title_length]
        score_cand_abstract = score_candidate_concat_input[
            :, self.config.max_title_length:self.config.max_title_length + self.config.max_abstract_length]
        score_cand_category = score_candidate_concat_input[
            :, self.config.max_title_length + self.config.max_abstract_length:self.config.max_title_length + self.config.max_abstract_length + 1]
        score_cand_subcategory = score_candidate_concat_input[
            :, self.config.max_title_length + self.config.max_abstract_length + 1:]

        # Reshape category/subcategory to remove the last dimension
        score_cand_category = ops.squeeze(score_cand_category, axis=-1)
        score_cand_subcategory = ops.squeeze(score_cand_subcategory, axis=-1)

        score_history_inputs = {
            "title_tokens": score_hist_title,
            "abstract_tokens": score_hist_abstract,
            "category_id": score_hist_category,
            "subcategory_id": score_hist_subcategory,
        }

        score_candidate_inputs = {
            "title_tokens": score_cand_title,
            "abstract_tokens": score_cand_abstract,
            "category_id": ops.expand_dims(score_cand_category, axis=-1),
            "subcategory_id": ops.expand_dims(score_cand_subcategory, axis=-1),
        }

        scorer_output = self.scorer.score_single_candidate(score_history_inputs, score_candidate_inputs)

        scorer_model = keras.Model(
            inputs=[score_history_concat_input, score_candidate_concat_input],
            outputs=scorer_output,
            name="naml_scorer_model"
        )

        return training_model, scorer_model

    def _build_newsencoder(self) -> keras.Model:
        """Legacy method for backward compatibility - returns news encoder."""
        return self.news_encoder

    def _build_userencoder(self) -> keras.Model:
        """Legacy method for backward compatibility - returns user encoder."""
        return self.user_encoder

    def _build_graph_models(self) -> Tuple[keras.Model, keras.Model]:
        """Legacy method for backward compatibility - returns training and scorer models."""
        return self.training_model, self.scorer_model

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
        """Main forward pass of the NAML model.
        
        Routes to appropriate methods based on training mode and input format.
        
        Args:
            inputs: Dictionary with input tensors
            training: Whether in training mode
            
        Returns:
            Model predictions based on mode and input format
        """
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
        # Prepare history inputs
        history_inputs = {
            "title_tokens": inputs["hist_tokens"],
            "abstract_tokens": inputs["hist_abstract_tokens"],
            "category_id": ops.expand_dims(inputs["hist_category"], axis=-1),
            "subcategory_id": ops.expand_dims(inputs["hist_subcategory"], axis=-1),
        }

        # Prepare candidate inputs
        candidate_inputs = {
            "title_tokens": inputs["cand_tokens"],
            "abstract_tokens": inputs["cand_abstract_tokens"],
            "category_id": ops.expand_dims(inputs["cand_category"], axis=-1),
            "subcategory_id": ops.expand_dims(inputs["cand_subcategory"], axis=-1),
        }

        return self.scorer.score_training_batch(history_inputs, candidate_inputs, training=True)

    def _handle_multiple_candidates(self, inputs):
        """Handle multiple candidate scoring with sigmoid scores."""
        # Prepare history inputs
        history_inputs = {
            "title_tokens": inputs["hist_tokens"],
            "abstract_tokens": inputs["hist_abstract_tokens"],
            "category_id": ops.expand_dims(inputs["hist_category"], axis=-1),
            "subcategory_id": ops.expand_dims(inputs["hist_subcategory"], axis=-1),
        }

        # Prepare candidate inputs
        candidate_inputs = {
            "title_tokens": inputs["cand_tokens"],
            "abstract_tokens": inputs["cand_abstract_tokens"],
            "category_id": ops.expand_dims(inputs["cand_category"], axis=-1),
            "subcategory_id": ops.expand_dims(inputs["cand_subcategory"], axis=-1),
        }

        return self.scorer.score_multiple_candidates(history_inputs, candidate_inputs, training=False)

    def get_config(self):
        """Returns the configuration of the NAML model for serialization."""
        base_config = super().get_config()
        base_config.update({
            "max_title_length": self.config.max_title_length,
            "max_abstract_length": self.config.max_abstract_length,
            "embedding_size": self.config.embedding_size,
            "category_embedding_dim": self.config.category_embedding_dim,
            "subcategory_embedding_dim": self.config.subcategory_embedding_dim,
            "cnn_filter_num": self.config.cnn_filter_num,
            "cnn_kernel_size": self.config.cnn_kernel_size,
            "word_attention_query_dim": self.config.word_attention_query_dim,
            "view_attention_query_dim": self.config.view_attention_query_dim,
            "user_attention_query_dim": self.config.user_attention_query_dim,
            "dropout_rate": self.config.dropout_rate,
            "activation": self.config.activation,
            "max_history_length": self.config.max_history_length,
            "max_impressions_length": self.config.max_impressions_length,
            "process_user_id": self.config.process_user_id,
            "seed": self.config.seed,
        })
        return base_config

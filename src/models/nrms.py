from typing import Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

import keras
from keras import layers, ops

from .layers import AdditiveAttentionLayer
from .base import BaseModel


@dataclass
class NRMSConfig:
    """Configuration class for NRMS model parameters."""
    embedding_size: int = 300
    multiheads: int = 16
    head_dim: int = 16
    attention_hidden_dim: int = 200
    dropout_rate: float = 0.2
    seed: int = 42
    max_title_length: int = 50
    max_history_length: int = 50
    max_impressions_length: int = 5
    process_user_id: bool = False


class NewsEncoder(keras.Model):
    """News encoder component for NRMS.
    
    Processes news titles through word embeddings, dropout, multi-head self-attention,
    and additive attention to produce news representations.
    """

    def __init__(self, config: NRMSConfig, embedding_layer: layers.Embedding, name: str = "news_encoder"):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the news encoder.
        
        Args:
            input_shape: Input shape (batch_size, title_length)
            
        Returns:
            Output shape (batch_size, embedding_size)
        """
        return (input_shape[0], self.config.embedding_size)

    def call(self, inputs, training=None):
        """Forward pass for news encoding.
        
        Args:
            inputs: News token sequences (batch_size, title_length)
            training: Whether in training mode
            
        Returns:
            News representations (batch_size, embedding_size)
        """
        # Word Embedding
        embedded_sequences = self.embedding_layer(inputs)

        # Dropout after embedding
        y = layers.Dropout(self.config.dropout_rate, seed=self.config.seed)(embedded_sequences, training=training)

        # Create padding mask for attention
        padding_mask = ops.not_equal(inputs, 0)

        # Multi-Head Self-Attention over words
        y = layers.MultiHeadAttention(
            num_heads=self.config.multiheads,
            key_dim=self.config.head_dim,
            dropout=self.config.dropout_rate,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.config.seed),
            name="title_word_self_attention",
        )(y, y, y, key_mask=padding_mask, training=training)

        # Apply dropout after self-attention
        y = layers.Dropout(self.config.dropout_rate, seed=self.config.seed)(y, training=training)

        # Additive Attention to get single news vector
        news_representation = AdditiveAttentionLayer(
            query_vec_dim=self.config.attention_hidden_dim,
            seed=self.config.seed,
            name="title_additive_attention",
        )(y, mask=padding_mask)

        return news_representation


class UserEncoder(keras.Model):
    """User encoder component for NRMS.
    
    Processes user history through news encoder, multi-head self-attention,
    and additive attention to produce user representations.
    """

    def __init__(self, config: NRMSConfig, news_encoder: NewsEncoder, name: str = "user_encoder"):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the user encoder.
        
        Args:
            input_shape: Input shape (batch_size, history_length, title_length)
            
        Returns:
            Output shape (batch_size, embedding_size)
        """
        return (input_shape[0], self.config.embedding_size)

    def call(self, inputs, training=None):
        """Forward pass for user encoding.
        
        Args:
            inputs: User history token sequences (batch_size, history_length, title_length)
            training: Whether in training mode
            
        Returns:
            User representations (batch_size, embedding_size)
        """
        # Apply news encoder to each news item in history
        click_title_presents = layers.TimeDistributed(
            self.news_encoder, name="time_distributed_news_encoder"
        )(inputs, training=training)

        # Create mask for history items
        history_mask = ops.any(ops.not_equal(inputs, 0), axis=-1)

        # Self-Attention over browsed news
        y = layers.MultiHeadAttention(
            num_heads=self.config.multiheads,
            key_dim=self.config.head_dim,
            dropout=self.config.dropout_rate,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.config.seed),
            name="browsed_news_self_attention",
        )(click_title_presents, click_title_presents, click_title_presents,
          key_mask=history_mask, training=training)

        # Additive Attention to get single user vector
        user_representation = AdditiveAttentionLayer(
            query_vec_dim=self.config.attention_hidden_dim,
            seed=self.config.seed,
            name="user_additive_attention",
        )(y, mask=history_mask)

        return user_representation


class NRMSScorer(keras.Model):
    """Scoring component for NRMS.
    
    Handles different scoring scenarios: training (multiple candidates with softmax),
    single candidate scoring (with sigmoid), and multiple candidate scoring (raw scores).
    """

    def __init__(self, config: NRMSConfig, news_encoder: NewsEncoder, user_encoder: UserEncoder,
                 name: str = "nrms_scorer"):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder

    def build(self, input_shape):
        super().build(input_shape)

    def score_training_batch(self, history_tokens, candidate_tokens, training=None):
        """Score training batch with softmax output."""
        user_repr = self.user_encoder(history_tokens, training=training)

        # Process candidates using TimeDistributed
        candidate_repr = layers.TimeDistributed(
            self.news_encoder, name="td_candidate_news_encoder"
        )(candidate_tokens, training=training)

        # Calculate scores using dot product
        scores = layers.Dot(axes=-1, name="dot_product_train")([candidate_repr, user_repr])

        # Apply softmax for training
        return layers.Activation("softmax", name="softmax_activation")(scores)

    def score_single_candidate(self, history_tokens, candidate_tokens, training=None):
        """Score single candidate with sigmoid output."""
        user_repr = self.user_encoder(history_tokens, training=training)
        candidate_repr = self.news_encoder(candidate_tokens, training=training)

        # Calculate score using dot product
        score = layers.Dot(axes=-1, name="dot_product_single")([candidate_repr, user_repr])

        # Apply sigmoid for probability
        return layers.Activation("sigmoid", name="sigmoid_activation")(score)

    def score_multiple_candidates(self, history_tokens, candidate_tokens, training=False):
        """Score multiple candidates using scorer model for consistency.
        
        This method processes each candidate individually through the scorer model
        to maintain architectural consistency and ensure sigmoid activation is applied.
        
        Args:
            history_tokens: User history tokens, shape (batch_size, history_len, title_len)
            candidate_tokens: Candidate tokens, shape (batch_size, num_candidates, title_len)
            training: Whether in training mode
            
        Returns:
            Scores with sigmoid activation applied, shape (batch_size, num_candidates)
        """
        batch_size = ops.shape(history_tokens)[0]
        num_candidates = ops.shape(candidate_tokens)[1]

        all_scores = []

        # Process each item in the batch
        for i in range(batch_size):
            # Get history for current item
            current_history = ops.expand_dims(
                history_tokens[i], 0
            )  # Shape: (1, history_len, title_len)

            # Score each candidate against this history
            candidate_scores = []
            for j in range(num_candidates):
                current_candidate = ops.expand_dims(
                    candidate_tokens[i, j], 0
                )  # Shape: (1, title_len)

                # Use scorer model which includes sigmoid activation
                score = self.score_single_candidate(
                    current_history, current_candidate, training=training
                )
                candidate_scores.append(score)

            # Combine scores for all candidates of this item
            item_scores = ops.concatenate(
                candidate_scores, axis=1
            )  # Shape: (1, num_candidates)
            all_scores.append(item_scores)

        # Combine scores for all items in batch
        return ops.concatenate(all_scores, axis=0)  # Shape: (batch_size, num_candidates)


class NRMS(BaseModel):
    """Neural News Recommendation with Multi-Head Self-Attention (NRMS) model.
    
    Refactored with clean architecture using separate components for better
    maintainability, testability, and code organization.
    """

    def __init__(
            self,
            processed_news: Dict[str, Any],
            embedding_size: int = 300,
            multiheads: int = 16,
            head_dim: int = 16,
            attention_hidden_dim: int = 200,
            dropout_rate: float = 0.2,
            seed: int = 42,
            max_title_length: int = 50,
            max_history_length: int = 50,
            max_impressions_length: int = 5,
            process_user_id: bool = False,
            name: str = "nrms",
            **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Create configuration object
        self.config = NRMSConfig(
            embedding_size=embedding_size,
            multiheads=multiheads,
            head_dim=head_dim,
            attention_hidden_dim=attention_hidden_dim,
            dropout_rate=dropout_rate,
            seed=seed,
            max_title_length=max_title_length,
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
        self.newsencoder = self.news_encoder
        self.userencoder = self.user_encoder
        self.process_user_id = process_user_id
        self.float_dtype = "float32"

    def build(self, input_shape):
        """Build the NRMS model.
        
        This method is called automatically by Keras to build the model layers.
        Since our components are already created in __init__, we just need to
        call the parent build method.
        
        Args:
            input_shape: Input shape (can be None for models with multiple inputs)
        """
        # Mark this layer as built
        super().build(input_shape)

    def _validate_processed_news(self) -> None:
        """Validate processed news data integrity."""
        required_keys = ["vocab_size", "embeddings"]
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
        # Create shared embedding layer
        self.embedding_layer = layers.Embedding(
            input_dim=self.processed_news["vocab_size"],
            output_dim=self.config.embedding_size,
            embeddings_initializer=keras.initializers.Constant(self.processed_news["embeddings"]),
            trainable=True,
            mask_zero=True,
            name="word_embedding",
        )

        # Create component encoders
        self.news_encoder = NewsEncoder(self.config, self.embedding_layer)
        self.user_encoder = UserEncoder(self.config, self.news_encoder)
        self.scorer = NRMSScorer(self.config, self.news_encoder, self.user_encoder)

        # Build training and scorer models for compatibility
        self.training_model, self.scorer_model = self._build_compatibility_models()

    def _build_compatibility_models(self) -> Tuple[keras.Model, keras.Model]:
        """Build training and scorer models for backward compatibility."""
        # ----- Training model -----
        history_input = keras.Input(
            shape=(self.config.max_history_length, self.config.max_title_length),
            dtype="int32", name="hist_tokens"
        )
        candidates_input = keras.Input(
            shape=(self.config.max_impressions_length, self.config.max_title_length),
            dtype="int32", name="cand_tokens"
        )

        training_output = self.scorer.score_training_batch(history_input, candidates_input)
        training_model = keras.Model(
            inputs=[history_input, candidates_input],
            outputs=training_output,
            name="nrms_training_model"
        )

        # ----- Scorer model -----
        history_input_score = keras.Input(
            shape=(self.config.max_history_length, self.config.max_title_length),
            dtype="int32", name="history_tokens_score"
        )
        single_candidate_input = keras.Input(
            shape=(self.config.max_title_length,),
            dtype="int32", name="single_candidate_tokens_score"
        )

        scorer_output = self.scorer.score_single_candidate(history_input_score, single_candidate_input)
        scorer_model = keras.Model(
            inputs=[history_input_score, single_candidate_input],
            outputs=scorer_output,
            name="nrms_scorer_model"
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
            # Training mode expects hist_tokens and cand_tokens
            required_keys = {"hist_tokens", "cand_tokens"}
            if not required_keys.issubset(input_keys):
                raise ValueError(
                    f"Training mode requires keys: {required_keys}, "
                    f"but got keys: {list(input_keys)}"
                )
        else:
            # Inference mode can have different formats
            valid_combinations = [
                {"hist_tokens", "cand_tokens"},  # Training format for validation
                {"history_tokens", "single_candidate_tokens"},  # Single candidate scoring
            ]

            if not any(combination.issubset(input_keys) for combination in valid_combinations):
                raise ValueError(
                    f"Inference mode expects one of: {valid_combinations}, "
                    f"but got keys: {list(input_keys)}"
                )

    def call(self, inputs, training=None):
        """Main forward pass of the NRMS model.
        
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
            return self._handle_training(inputs, training)
        else:
            # Inference mode: route based on input format
            if "single_candidate_tokens" in inputs:
                return self._handle_single_candidate(inputs)
            elif "hist_tokens" in inputs and "cand_tokens" in inputs:
                return self._handle_multiple_candidates(inputs)
            else:
                raise ValueError("Invalid input format for inference mode")

    def _handle_training(self, inputs, training=None):
        """Handle training batch scoring with softmax output."""
        # Extract the correct keys based on input format
        history_key = "hist_tokens"
        candidates_key = "cand_tokens"

        return self.scorer.score_training_batch(
            inputs[history_key], inputs[candidates_key], training=training
        )

    def _handle_single_candidate(self, inputs, training=None):
        """Handle single candidate scoring with sigmoid output."""
        return self.scorer.score_single_candidate(
            inputs["history_tokens"], inputs["single_candidate_tokens"], training=training
        )

    def _handle_multiple_candidates(self, inputs):
        """Handle multiple candidate scoring with sigmoid scores."""
        candidates_key = "cand_tokens"
        history_key = "hist_tokens"

        return self.scorer.score_multiple_candidates(
            inputs[history_key], inputs[candidates_key], training=False
        )

    def get_config(self):
        """Returns the configuration of the NRMS model for serialization."""
        base_config = super().get_config()
        base_config.update({
            "embedding_size": self.config.embedding_size,
            "multiheads": self.config.multiheads,
            "head_dim": self.config.head_dim,
            "attention_hidden_dim": self.config.attention_hidden_dim,
            "dropout_rate": self.config.dropout_rate,
            "seed": self.config.seed,
            "max_title_length": self.config.max_title_length,
            "max_history_length": self.config.max_history_length,
            "max_impressions_length": self.config.max_impressions_length,
            "process_user_id": self.config.process_user_id,
        })
        return base_config

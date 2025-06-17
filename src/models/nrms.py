from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

# Assuming AdditiveSelfAttention is correctly defined in .layers
from .layers import AdditiveSelfAttention
from .base import BaseModel
import numpy as np

# from utils.losses import get_loss # Not used in this file directly
from utils.saving import save_predictions_to_file_fn


class SelfAttention(layers.Layer):
    """Multi-head self attention layer using Keras's built-in implementation.

    This layer is a wrapper around Keras's MultiHeadAttention that simplifies self-attention usage.
    Key differences from raw MultiHeadAttention:
    1. Simplified Interface:
        - Automatically uses same tensor for query, key, and value (self-attention)
        - No need to manually specify q,k,v when they're the same tensor

    2. Fixed Output Shape:
        - Ensures output matches input dimensions
        - Handles dimension management internally

    3. Built-in Features:
        - Includes dropout for regularization
        - Handles layer normalization
        - Manages attention head dimensions

    Usage in NRMS:
    1. Word-level attention: Captures relationships between words in news titles
    2. News-level attention: Models relationships between news articles in user history

    Example:
        # For word embeddings in a title
        word_embeddings = ... # shape: (batch_size, seq_len, embed_dim)
        attention_output = SelfAttention(multiheads=8, head_dim=64)(word_embeddings)
        # Output shape: (batch_size, seq_len, multiheads * head_dim)
    """

    def __init__(self, multiheads, head_dim, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.seed = seed
        self.multihead_attention = None

    def build(self, input_shape):
        """Initializes the internal MultiHeadAttention layer.

        This method creates the MultiHeadAttention layer with the specified number of heads, head dimension, and seed.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Create the MultiHeadAttention layer
        self.multihead_attention = layers.MultiHeadAttention(
            num_heads=self.multiheads,
            key_dim=self.head_dim,
            value_dim=self.head_dim,  # value_dim should typically match key_dim
            output_shape=self.head_dim
            * self.multiheads,  # output_shape is inferred, not set directly. This might be an old API usage or custom expectation.
            # For standard MHA, output is (batch_size, seq_len, num_heads * key_dim) if output_shape is None.
            # If output_shape is specified, it dictates the last dimension.
            seed=self.seed,
        )
        super().build(input_shape)

    def call(
        self, inputs, training=None
    ):  # Added training argument for consistency, MHA handles it
        """Applies multi-head self-attention to the input tensor(s).

        This method computes self-attention over the input(s) using the underlying Keras MultiHeadAttention layer.

        Args:
            inputs: Either a single tensor (used as query, key, and value) or a list of three tensors [query, key, value].
            training: Boolean or None. Indicates whether the layer should behave in training mode or inference mode.

        Returns:
            tf.Tensor: The result of the multi-head attention operation.
        """
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        # Pass the training argument to the underlying MHA layer
        return self.multihead_attention(
            query, value, key, training=training
        )  # Correct order for MHA is query, value, key

    def get_config(self):
        """Returns the configuration of the SelfAttention layer for serialization.

        This method provides a dictionary of the layer's hyperparameters and settings,
        enabling the layer to be saved and reconstructed later.

        Returns:
            dict: Layer configuration including number of heads, head dimension, and seed.
        """
        config = super().get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "seed": self.seed,
            }
        )
        return config


class NRMS(BaseModel):
    """Neural News Recommendation with Multi-Head Self-Attention (NRMS) model."""

    def __init__(
        self,
        processed_news,
        embedding_size=300,
        multiheads=16,
        head_dim=16,
        attention_hidden_dim=200,
        dropout_rate=0.2,
        seed=42,
        name="nrms",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.embedding_size = embedding_size
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout_rate = dropout_rate
        self.seed = seed

        tf.random.set_seed(
            self.seed
        )  # Set global TF seed for reproducibility if needed elsewhere too

        self.vocab_size = processed_news["vocab_size"]
        self.embeddings_matrix = processed_news["embeddings"]

        # Build components
        self.embedding_layer = layers.Embedding(
            self.vocab_size,
            self.embedding_size,
            embeddings_initializer=tf.keras.initializers.Constant(self.embeddings_matrix),
            trainable=True,
            mask_zero=False,
            name="word_embedding",
        )

        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder()

        # Build the main training model and the scorer model
        self.training_model, self.scorer_model = self._build_graph_models()

    def _build_newsencoder(self) -> tf.keras.Model:
        """Build the news encoder component.

        The news encoder processes news titles through:
        1. Word embeddings
        2. Dropout for regularization
        3. Word-level self-attention to capture important words
        4. Additive attention to get a single news vector

        Returns:
            tf.keras.Model: The news encoder model
                Input: (batch_size, title_length) - tokenized news titles
                Output: (batch_size, embedding_dim) - news embeddings
        """
        sequences_input_title = tf.keras.Input(
            shape=(None,), dtype="int32", name="news_tokens_input"
        )

        # Word Embedding
        embedded_sequences_title = self.embedding_layer(sequences_input_title)

        # Dropout after embedding
        y = layers.Dropout(self.dropout_rate, seed=self.seed)(embedded_sequences_title)

        # Self-Attention over words in a title
        y = SelfAttention(
            self.multiheads, self.head_dim, seed=self.seed, name="title_word_self_attention"
        )([y, y, y])
        y = layers.Dropout(self.dropout_rate, seed=self.seed + 1)(y)

        # Additive Attention to get a single news vector
        pred_title = AdditiveSelfAttention(
            self.attention_hidden_dim, name="title_additive_attention"
        )(y)

        return tf.keras.Model(sequences_input_title, pred_title, name="news_encoder")

    def _build_userencoder(self) -> tf.keras.Model:
        """Build the user encoder component.

        The user encoder processes user history through:
        1. News encoder to get embeddings for each news in history
        2. News-level self-attention to model user interests
        3. Additive attention to get a single user vector

        Returns:
            tf.keras.Model: The user encoder model
                Input: (batch_size, history_length, title_length) - tokenized news history
                Output: (batch_size, embedding_dim) - user embeddings
        """
        # Input shape: (batch_size, num_history_news, num_words_in_title)
        his_input_title_tokens = tf.keras.Input(
            shape=(None, None), dtype="int32", name="history_news_tokens_input"
        )

        # Apply newsencoder to each news item in history
        click_title_presents = layers.TimeDistributed(
            self.newsencoder, name="time_distributed_news_encoder"
        )(his_input_title_tokens)

        # Self-Attention over browsed news
        y = SelfAttention(
            self.multiheads, self.head_dim, seed=self.seed + 2, name="browsed_news_self_attention"
        )([click_title_presents, click_title_presents, click_title_presents])
        y = layers.Dropout(self.dropout_rate, seed=self.seed + 3)(y)

        # Additive Attention to get a single user vector
        user_present = AdditiveSelfAttention(
            self.attention_hidden_dim, name="user_additive_attention"
        )(y)

        return tf.keras.Model(his_input_title_tokens, user_present, name="user_encoder")

    def _build_graph_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Build the training and scoring models.

        This method builds two models:
        1. Training Model:
            - Takes user history and multiple candidate news
            - Outputs softmax probabilities for each candidate

        2. Scorer Model:
            - Takes user history and a single candidate news
            - Outputs a sigmoid probability for the candidate

        Returns:
            Tuple containing:
                - Training model
                - Scorer model
        """
        # --- Inputs for Training Model ---
        history_tokens_input_train = tf.keras.Input(
            shape=(None, None), dtype="int32", name="history_tokens_train"
        )
        candidate_tokens_input_train = tf.keras.Input(
            shape=(None, None), dtype="int32", name="candidate_tokens_train"
        )

        # ------ Training Model Graph ------
        # Get user representation from history
        user_representation_train = self.userencoder(history_tokens_input_train)

        # Get candidate news representations
        candidate_news_representation_train = layers.TimeDistributed(
            self.newsencoder, name="td_candidate_news_encoder_train"
        )(candidate_tokens_input_train)

        # Calculate scores using Dot layer
        scores = layers.Dot(axes=-1, name="dot_product_train")(
            [candidate_news_representation_train, user_representation_train]
        )

        # Apply softmax for training
        preds_train = layers.Activation("softmax", name="softmax_activation_train")(scores)

        training_model = tf.keras.Model(
            inputs=[history_tokens_input_train, candidate_tokens_input_train],
            outputs=preds_train,
            name="nrms_training_model",
        )

        # ------ Inputs for Scorer Model ------
        history_tokens_input_score = tf.keras.Input(
            shape=(None, None), dtype="int32", name="history_tokens_score"
        )
        single_candidate_tokens_input_score = tf.keras.Input(
            shape=(None,), dtype="int32", name="single_candidate_tokens_score"
        )

        # --- Scorer Model Graph ---
        user_representation_score = self.userencoder(history_tokens_input_score)
        single_candidate_representation_score = self.newsencoder(
            single_candidate_tokens_input_score
        )

        # Calculate score using Dot layer
        pred_score = layers.Dot(axes=-1, name="dot_product_score")(
            [single_candidate_representation_score, user_representation_score]
        )

        # Apply sigmoid for single prediction probability
        pred_score = layers.Activation("sigmoid", name="sigmoid_activation_score")(pred_score)

        scorer_model = tf.keras.Model(
            inputs=[history_tokens_input_score, single_candidate_tokens_input_score],
            outputs=pred_score,
            name="nrms_scorer_model",
        )

        return training_model, scorer_model

    def call(self, inputs, training=None):
        """Main forward pass of the NRMS model.

        This method serves as the main entry point for both training and inference. It handles three different
        input formats and routes them to the appropriate internal model (training_model or scorer_model).

        Input Formats:
        1. Training: {'hist_tokens': history, 'cand_tokens': candidates}
            - Used during model training
            - history: shape (batch_size, history_len, title_len)
            - candidates: shape (batch_size, num_candidates, title_len)

        2. Single candidate scoring: {'history_tokens': history, 'single_candidate_tokens': candidate}
            - Used for scoring a single news article
            - history: shape (batch_size, history_len, title_len)
            - candidate: shape (batch_size, title_len)

        3. Multiple candidates scoring: {'history_tokens': history, 'candidate_tokens': candidates}
            - Used for scoring multiple news articles
            - history: shape (batch_size, history_len, title_len)
            - candidates: shape (batch_size, num_candidates, title_len)

        Args:
            inputs (dict): Dictionary containing input tensors in one of the three formats described above
            training (bool, optional): Whether the model is in training mode. Defaults to None.

        Returns:
            tf.Tensor: Model predictions
                - For training: shape (batch_size, num_candidates) with softmax probabilities
                - For single candidate: shape (batch_size, 1) with sigmoid probabilities
                - For multiple candidates: shape (batch_size, num_candidates) with raw scores

        Raises:
            ValueError: If input format is invalid for the current mode (training/inference)
        """
        # Training mode - use training model
        if training:
            return self._handle_training(inputs)

        # Inference mode - use scorer model
        return self._handle_inference(inputs)

    def _handle_training(self, inputs):
        """Handle the forward pass during training mode.

        This method processes inputs for the training model, which expects a list of two tensors:
        [history_tokens, candidate_tokens]. The training model outputs probabilities for each
        candidate in the batch.

        Args:
            inputs (dict): Dictionary containing:
                - 'hist_tokens': User history tokens, shape (batch_size, history_len, title_len)
                - 'cand_tokens': Candidate news tokens, shape (batch_size, num_candidates, title_len)

        Returns:
            tf.Tensor: Softmax probabilities for each candidate
                Shape: (batch_size, num_candidates)
        """
        # Convert dict inputs to list format expected by training model
        history = inputs["hist_tokens"]
        candidates = inputs["cand_tokens"]
        return self.training_model([history, candidates], training=True)

    def _handle_inference(self, inputs):
        """Handle the forward pass during inference mode.

        This method routes the input to either single candidate scoring or multiple candidate scoring
        based on the input format. It uses the scorer_model which is optimized for inference.

        Args:
            inputs (dict): Dictionary containing either:
                - 'history_tokens' and 'single_candidate_tokens' for single candidate scoring
                - 'history_tokens' and 'candidate_tokens' for multiple candidate scoring

        Returns:
            tf.Tensor: Model predictions
                - For single candidate: shape (batch_size, 1) with sigmoid probabilities
                - For multiple candidates: shape (batch_size, num_candidates) with raw scores

        Raises:
            ValueError: If input format is invalid for inference
        """
        # Case 1: Single candidate scoring
        if "single_candidate_tokens" in inputs:
            history = inputs["history_tokens"]
            candidate = inputs["single_candidate_tokens"]
            return self.scorer_model([history, candidate], training=False)

        # Case 2: Multiple candidates scoring
        if "cand_tokens" in inputs:
            return self._score_multiple_candidates(inputs)

        raise ValueError(
            "Invalid input format for inference. Expected 'single_candidate_tokens' or 'cand_tokens'"
        )

    def _score_multiple_candidates(self, inputs):
        """Score multiple candidates for each history in the batch.

        This method processes a batch of user histories and their corresponding candidate news articles.
        For each user history, it scores all candidates against that history using the scorer_model.
        The scoring is done one candidate at a time to maintain batch consistency.

        Args:
            inputs (dict): Dictionary containing:
                - 'hist_tokens': User history tokens, shape (batch_size, history_len, title_len)
                - 'cand_tokens': Candidate news tokens, shape (batch_size, num_candidates, title_len)

        Returns:
            tf.Tensor: Scores for all candidates
                Shape: (batch_size, num_candidates)
                Each row contains the scores for all candidates for one user history

        Note:
            This method processes candidates one at a time to ensure proper batch handling
            and to maintain consistency with the scorer_model's expectations.
        """
        history_batch = inputs["hist_tokens"]  # Shape: (batch_size, history_len, title_len)
        candidates_batch = inputs["cand_tokens"]  # Shape: (batch_size, num_candidates, title_len)

        batch_size = tf.shape(history_batch)[0]
        num_candidates = tf.shape(candidates_batch)[1]

        all_scores = []

        # Process each item in the batch
        for i in range(batch_size):
            # Get history for current item
            current_history = tf.expand_dims(
                history_batch[i], 0
            )  # Shape: (1, history_len, title_len)

            # Score each candidate against this history
            candidate_scores = []
            for j in range(num_candidates):
                current_candidate = tf.expand_dims(
                    candidates_batch[i, j], 0
                )  # Shape: (1, title_len)
                score = self.scorer_model([current_history, current_candidate], training=False)
                candidate_scores.append(score)

            # Combine scores for all candidates of this item
            item_scores = tf.concat(candidate_scores, axis=1)  # Shape: (1, num_candidates)
            all_scores.append(item_scores)

        # Combine scores for all items in batch
        return tf.concat(all_scores, axis=0)  # Shape: (batch_size, num_candidates)

    def get_config(self):
        """Returns the configuration of the NRMS model for serialization.

        This method provides a dictionary of the model's hyperparameters and settings,
        enabling the model to be saved and reconstructed later.

        Returns:
            dict: Model configuration including embedding size, attention parameters, dropout rate, seed, and vocabulary size.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_size": self.embedding_size,
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "attention_hidden_dim": self.attention_hidden_dim,
                "dropout_rate": self.dropout_rate,
                "seed": self.seed,
                "vocab_size": self.vocab_size,
                # "embeddings_matrix": self.embeddings_matrix.tolist() # Careful with large matrices in config
            }
        )
        # Note: processed_news is not part of the config as it's data.
        # Keras will save/load weights of sub-models (embedding_layer, newsencoder, userencoder, training_model, scorer_model)
        return config

from typing import Tuple, Dict, Any
import numpy as np

# Set JAX backend for Keras 3

import keras
from keras import layers, ops

from .layers import AdditiveAttentionLayer
from .base import BaseModel


class NRMS(BaseModel):
    """Neural News Recommendation with Multi-Head Self-Attention (NRMS) model."""

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
            process_user_id: bool = False,  # Only used in base model
            name: str = "nrms",
            **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.embedding_size = embedding_size
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout_rate = dropout_rate
        self.process_user_id = process_user_id
        self.seed = seed
        self.max_title_length = max_title_length
        self.max_history_length = max_history_length
        self.max_impressions_length = max_impressions_length

        self.vocab_size = processed_news["vocab_size"]
        self.embeddings_matrix = processed_news["embeddings"]
        
        # DEBUG: Check embeddings matrix for NaN
        import numpy as np
        if np.isnan(self.embeddings_matrix).any():
            print(f"WARNING: Embeddings matrix contains NaN values!")
            print(f"Number of NaN values: {np.isnan(self.embeddings_matrix).sum()}")
            nan_indices = np.where(np.isnan(self.embeddings_matrix))[0]
            print(f"NaN at token indices: {nan_indices[:10]}...")  # Show first 10
        else:
            print(f"INFO: Embeddings matrix is clean (no NaN)")
            print(f"      Shape: {self.embeddings_matrix.shape}")
            print(f"      Min: {np.min(self.embeddings_matrix):.6f}, Max: {np.max(self.embeddings_matrix):.6f}")
            # Check for padding token embedding
            padding_embedding = self.embeddings_matrix[0]  # Usually index 0 is padding
            print(f"      Padding token embedding norm: {np.linalg.norm(padding_embedding):.6f}")

        # Build components
        self.embedding_layer = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=keras.initializers.Constant(self.embeddings_matrix),
            trainable=True,
            mask_zero=True,  # Enable automatic masking for padding tokens
            name="word_embedding",
        )

        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder()

        # Build the main training model and the scorer model
        self.training_model, self.scorer_model = self._build_graph_models()
    

    def _build_newsencoder(self) -> keras.Model:
        """Build the news encoder component.

        The news encoder processes news titles through:
        1. Word embeddings
        2. Dropout for regularization
        3. Word-level self-attention to capture important words
        4. Additive attention to get a single news vector

        Returns:
            keras.Model: The news encoder model
                Input: (batch_size, title_length) - tokenized news titles
                Output: (batch_size, embedding_dim) - news embeddings
        """
        sequences_input_title = keras.Input(
            shape=(self.max_title_length,), dtype="int32", name="news_tokens_input"
        )
        
        # Word Embedding
        embedded_sequences_title = self.embedding_layer(sequences_input_title)

        # Dropout after embedding
        y = layers.Dropout(self.dropout_rate, seed=self.seed)(embedded_sequences_title)

        # Multi-Head Self-Attention over words in a title
        # Create padding mask for attention
        padding_mask = ops.not_equal(sequences_input_title, 0)  # True for non-padding
        
        y = layers.MultiHeadAttention(
            num_heads=self.multiheads,
            key_dim=self.head_dim,
            dropout=self.dropout_rate,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed),
            name="title_word_self_attention",
        )(y, y, y, key_mask=padding_mask)

        # Apply dropout after self-attention
        y = layers.Dropout(self.dropout_rate, seed=self.seed)(y)

        # Additive Attention to get a single news vector with masking
        pred_title = AdditiveAttentionLayer(
            query_vec_dim=self.attention_hidden_dim,
            seed=self.seed,
            name="title_additive_attention",
        )(y, mask=padding_mask)

        return keras.Model(sequences_input_title, pred_title, name="news_encoder")

    def _build_userencoder(self) -> keras.Model:
        """Build the user encoder component.

        The user encoder processes user history through:
        1. News encoder to get embeddings for each news in history
        2. News-level self-attention to model user interests
        3. Additive attention to get a single user vector

        Returns:
            keras.Model: The user encoder model
                Input: (batch_size, history_length, title_length) - tokenized news history
                Output: (batch_size, embedding_dim) - user embeddings
        """
        # Input shape: (batch_size, num_history_news, num_words_in_title)
        his_input_title_tokens = keras.Input(
            shape=(self.max_history_length, self.max_title_length),
            dtype="int32",
            name="history_news_tokens_input",
        )

        # Apply newsencoder to each news item in history
        click_title_presents = layers.TimeDistributed(
            self.newsencoder, name="time_distributed_news_encoder"
        )(his_input_title_tokens)

        # Create mask for history items (shape: batch_size, history_length)
        # Check if any token in each news item is non-zero
        history_mask = ops.any(ops.not_equal(his_input_title_tokens, 0), axis=-1)

        # Self-Attention over browsed news with key_mask
        y = layers.MultiHeadAttention(
            num_heads=self.multiheads,
            key_dim=self.head_dim,
            dropout=self.dropout_rate,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed),
            name="browsed_news_self_attention",
        )(click_title_presents, click_title_presents, click_title_presents, key_mask=history_mask)

        # Additive Attention to get a single user vector with explicit masking
        user_present = AdditiveAttentionLayer(
            query_vec_dim=self.attention_hidden_dim,
            seed=self.seed,
            name="user_additive_attention",
        )(y, mask=history_mask)

        return keras.Model(his_input_title_tokens, user_present, name="user_encoder")

    def _build_graph_models(self) -> Tuple[keras.Model, keras.Model]:
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
        history_tokens_input_train = keras.Input(
            shape=(self.max_history_length, self.max_title_length),
            dtype="int32",
            name="history_tokens_train",
        )
        candidate_tokens_input_train = keras.Input(
            shape=(self.max_impressions_length, self.max_title_length),
            dtype="int32",
            name="candidate_tokens_train",
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

        training_model = keras.Model(
            inputs=[history_tokens_input_train, candidate_tokens_input_train],
            outputs=preds_train,
            name="nrms_training_model",
        )

        # ------ Inputs for Scorer Model ------
        history_tokens_input_score = keras.Input(
            shape=(self.max_history_length, self.max_title_length),
            dtype="int32",
            name="history_tokens_score",
        )
        single_candidate_tokens_input_score = keras.Input(
            shape=(self.max_title_length,),
            dtype="int32",
            name="single_candidate_tokens_score",
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

        scorer_model = keras.Model(
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
            Tensor: Model predictions
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
            Tensor: Softmax probabilities for each candidate
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
            Tensor: Model predictions
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
        """Optimized multiple candidate scoring using JAX backend."""
        history_batch = inputs["hist_tokens"]
        candidates_batch = inputs["cand_tokens"]

        # Debug: print shapes
        print(f"DEBUG: history_batch.shape = {ops.shape(history_batch)}")
        print(f"DEBUG: candidates_batch.shape = {ops.shape(candidates_batch)}")

        # Get user representations for entire batch
        user_representations = self.userencoder(history_batch, training=False)

        # Debug: print user representations shape and check for NaN/Inf
        print(f"DEBUG: user_representations.shape = {ops.shape(user_representations)}")
        user_np = ops.convert_to_numpy(user_representations)
        print(f"DEBUG: user_representations has NaN: {np.isnan(user_np).any()}")
        print(f"DEBUG: user_representations has Inf: {np.isinf(user_np).any()}")
        print(f"DEBUG: user_representations min/max: {np.min(user_np):.6f} / {np.max(user_np):.6f}")

        # Process all candidates efficiently
        batch_size = ops.shape(candidates_batch)[0]
        num_candidates = ops.shape(candidates_batch)[1]

        # Reshape candidates for batch processing
        candidates_flat = ops.reshape(
            candidates_batch,
            (batch_size * num_candidates, self.max_title_length)
        )

        # Get candidate representations
        candidate_representations_flat = self.newsencoder(candidates_flat, training=False)

        # Reshape back to batch format
        candidate_representations = ops.reshape(
            candidate_representations_flat,
            (batch_size, num_candidates, self.embedding_size)
        )

        # Debug: print candidate representations shape and check for NaN/Inf
        print(f"DEBUG: candidate_representations.shape = {ops.shape(candidate_representations)}")
        print(f"DEBUG: candidate_representations_flat.shape = {ops.shape(candidate_representations_flat)}")
        print(f"DEBUG: batch_size = {batch_size}, num_candidates = {num_candidates}, embedding_size = {self.embedding_size}")
        
        candidate_np = ops.convert_to_numpy(candidate_representations)
        print(f"DEBUG: candidate_representations has NaN: {np.isnan(candidate_np).any()}")
        print(f"DEBUG: candidate_representations has Inf: {np.isinf(candidate_np).any()}")
        print(f"DEBUG: candidate_representations min/max: {np.min(candidate_np):.6f} / {np.max(candidate_np):.6f}")

        # Debug: print exact shapes before einsum
        print(f"DEBUG: About to do einsum with:")
        print(f"  candidate_representations.shape = {ops.shape(candidate_representations)}")
        print(f"  user_representations.shape = {ops.shape(user_representations)}")
        print(f"  Expected: candidate_representations (64, 5, 300), user_representations (64, 300)")

        # Compute scores using explicit matrix operations instead of einsum
        # Expand user_representations to match candidate dimensions for element-wise multiplication
        # user_representations: (batch_size, embedding_size) -> (batch_size, 1, embedding_size)
        user_expanded = ops.expand_dims(user_representations, axis=1)  # (64, 1, 300)

        # Element-wise multiplication: (64, 5, 300) * (64, 1, 300) -> (64, 5, 300)
        similarity = candidate_representations * user_expanded

        # Sum over the feature dimension to get scores: (64, 5, 300) -> (64, 5)
        scores = ops.sum(similarity, axis=-1)

        print(f"DEBUG: scores.shape = {ops.shape(scores)}")
        
        # Debug: check final scores for NaN/Inf
        scores_np = ops.convert_to_numpy(scores)
        print(f"DEBUG: final scores has NaN: {np.isnan(scores_np).any()}")
        print(f"DEBUG: final scores has Inf: {np.isinf(scores_np).any()}")
        print(f"DEBUG: final scores min/max: {np.min(scores_np):.6f} / {np.max(scores_np):.6f}")

        return scores

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
            }
        )
        # Note: processed_news is not part of the config as it's data.
        # Keras will save/load weights of sub-models (embedding_layer, newsencoder, userencoder, training_model, scorer_model)
        return config

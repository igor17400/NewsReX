from typing import Tuple, Dict, Any, Optional

import tensorflow as tf
from tensorflow.keras import layers

from .layers import AdditiveAttentionLayer
from .base import BaseModel


class LSTUR(BaseModel):
    """Neural News Recommendation with Long- and Short-term User Representations (LSTUR) model.

    This model is based on the paper: "Neural News Recommendation with Long- and Short-term User Representations"
    by D. An et al. It captures both long-term and short-term user interests using different neural architectures.

    Key features:
    - Long-term user representation: Uses user ID embeddings to capture stable preferences
    - Short-term user representation: Uses LSTM/GRU to model recent browsing behavior
    - News encoder: Processes news titles using CNN and attention mechanisms
    - Dual user modeling: Combines long-term and short-term representations
    """

    def __init__(
        self,
        processed_news: Dict[str, Any],
        embedding_size: int = 300,
        user_embedding_dim: int = 100,
        cnn_filter_num: int = 400,
        cnn_kernel_size: int = 3,
        attention_hidden_dim: int = 200,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        max_title_length: int = 30,
        max_history_length: int = 50,
        max_impressions_length: int = 5,
        num_users: int = 100000,  # Total number of users in the dataset
        user_representation_type: str = "lstm",  # "lstm" or "gru"
        process_user_id: bool = True,  # Whether to use user ID embeddings
        seed: int = 42,
        name: str = "lstur",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Configurable parameters
        self.embedding_size = embedding_size
        self.user_embedding_dim = user_embedding_dim
        self.cnn_filter_num = cnn_filter_num
        self.cnn_kernel_size = cnn_kernel_size
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.max_title_length = max_title_length
        self.max_history_length = max_history_length
        self.max_impressions_length = max_impressions_length
        self.num_users = num_users
        self.user_representation_type = user_representation_type.lower()
        self.process_user_id = process_user_id
        self.seed = seed

        tf.random.set_seed(self.seed)

        # Unpack processed data from the dataset
        self.vocab_size = processed_news["vocab_size"]
        self.embeddings_matrix = processed_news["embeddings"]

        # Validate user representation type
        if self.user_representation_type not in ["lstm", "gru"]:
            raise ValueError("user_representation_type must be 'lstm' or 'gru'")

        # Build components
        self.word_embedding_layer = self._build_word_embedding_layer()
        if self.process_user_id:
            self.user_embedding_layer = self._build_user_embedding_layer()
        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder()

        # Build final training and scoring models
        self.training_model, self.scorer_model = self._build_graph_models()

    def _build_word_embedding_layer(self) -> layers.Embedding:
        """Builds the word embedding layer for news titles."""
        return layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.Constant(self.embeddings_matrix),
            trainable=True,
            mask_zero=False,
            name="word_embedding",
        )

    def _build_user_embedding_layer(self) -> layers.Embedding:
        """Builds the user embedding layer for long-term user representation."""
        return layers.Embedding(
            input_dim=self.num_users + 1,  # +1 for padding/unknown users
            output_dim=self.user_embedding_dim,
            trainable=True,
            mask_zero=False,
            name="user_embedding",
        )

    def _build_newsencoder(self) -> tf.keras.Model:
        """Builds the news encoder which processes news titles."""
        input_title = tf.keras.Input(
            shape=(self.max_title_length,), dtype="int32", name="news_tokens"
        )

        # Word embedding
        embedded_title = self.word_embedding_layer(input_title)
        title_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)(embedded_title)

        # CNN for feature extraction
        title_cnn = layers.Conv1D(
            self.cnn_filter_num,
            self.cnn_kernel_size,
            activation=self.activation,
            padding="same",
            name="title_cnn",
        )(title_dropout)

        title_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)(title_cnn)

        # Additive attention to get a single news vector
        title_attention = AdditiveAttentionLayer(
            self.attention_hidden_dim, name="title_word_attention"
        )(title_dropout)

        return tf.keras.Model(input_title, title_attention, name="news_encoder")

    def _build_userencoder(self) -> tf.keras.Model:
        """Builds the user encoder which combines long-term and short-term representations."""
        # Input for news history (short-term representation)
        history_input = tf.keras.Input(
            shape=(self.max_history_length, self.max_title_length),
            dtype="int32",
            name="history_tokens",
        )
        
        user_ids = tf.keras.Input(
            shape=(1,),
            dtype="int32",
            name="user_ids",
        )

        # Short-term user representation
        # Process each news in history
        news_vectors = layers.TimeDistributed(self.newsencoder, name="td_history_news_encoder")(
            history_input
        )

        # Apply LSTM or GRU for sequential modeling
        if self.user_representation_type == "lstm":
            short_term_repr = layers.LSTM(
                self.cnn_filter_num,
                return_sequences=False,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name="history_lstm",
            )(news_vectors)
        else:  # GRU
            short_term_repr = layers.GRU(
                self.cnn_filter_num,
                return_sequences=False,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name="history_gru",
            )(news_vectors)

        if self.process_user_id: # Long-term user representation
            # Long-term user representation using extracted user IDs
            user_embedding = self.user_embedding_layer(user_ids)
            user_embedding = layers.Reshape((self.user_embedding_dim,))(user_embedding)

            # Project user embedding to match news vector dimension
            user_projection = layers.Dense(
                self.cnn_filter_num,
                activation=self.activation,
                name="user_projection",
            )(user_embedding)

            # Combine representations (addition)
            combined_repr = layers.Add(name="combine_user_repr")([short_term_repr, user_projection])

            return tf.keras.Model([history_input, user_ids], combined_repr, name="user_encoder")

        # Only use short-term representation (LSTM/GRU output)
        return tf.keras.Model([history_input, user_ids], short_term_repr, name="user_encoder")

    def _build_graph_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Builds the main training and scoring models."""
        # --- Inputs for Training Model ---
        history_tokens_input_train = tf.keras.Input(
            shape=(self.max_history_length, self.max_title_length),
            dtype="int32",
            name="history_tokens_train",
        )
        user_ids = tf.keras.Input(
            shape=(1,),  # Single user ID per batch item
            dtype="int32",
            name="user_ids_train",
        )
        candidate_tokens_input_train = tf.keras.Input(
            shape=(self.max_impressions_length, self.max_title_length),
            dtype="int32",
            name="candidate_tokens_train",
        )

        # --- Training Model Graph ---
        # Pass history tokens and user ID separately to userencoder
        user_representation_train = self.userencoder([history_tokens_input_train, user_ids])

        # Get candidate news representations (no user IDs needed for news encoding)
        candidate_news_representation_train = layers.TimeDistributed(
            self.newsencoder, name="td_candidate_news_encoder_train"
        )(candidate_tokens_input_train)

        # Calculate scores using dot product
        scores = layers.Dot(axes=-1, name="dot_product_train")(
            [candidate_news_representation_train, user_representation_train]
        )

        # Apply softmax for training
        preds_train = layers.Activation("softmax", name="softmax_activation_train")(scores)

        training_model = tf.keras.Model(
            inputs=[history_tokens_input_train, user_ids, candidate_tokens_input_train],
            outputs=preds_train,
            name="lstur_training_model",
        )

        # --- Inputs for Scorer Model ---
        history_tokens_input_score = tf.keras.Input(
            shape=(self.max_history_length, self.max_title_length),
            dtype="int32",
            name="history_tokens_score",
        )
        user_ids_score = tf.keras.Input(
            shape=(1,),  # Single user ID per batch item
            dtype="int32",
            name="user_ids_score",
        )
        single_candidate_tokens_input_score = tf.keras.Input(
            shape=(self.max_title_length,),
            dtype="int32",
            name="single_candidate_tokens_score",
        )

        # --- Scorer Model Graph ---
        # Pass history tokens and user ID separately to userencoder
        user_representation_score = self.userencoder([history_tokens_input_score, user_ids_score])
        single_candidate_representation_score = self.newsencoder(
            single_candidate_tokens_input_score
        )

        # Calculate score using dot product
        pred_score = layers.Dot(axes=-1, name="dot_product_score")(
            [single_candidate_representation_score, user_representation_score]
        )

        # Apply sigmoid for single prediction probability
        pred_score = layers.Activation("sigmoid", name="sigmoid_activation_score")(pred_score)

        scorer_model = tf.keras.Model(
            inputs=[history_tokens_input_score, user_ids_score, single_candidate_tokens_input_score],
            outputs=pred_score,
            name="lstur_scorer_model",
        )

        return training_model, scorer_model

    def call(self, inputs: Dict[str, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """Main forward pass of the LSTUR model.

        This method serves as the main entry point for both training and inference.
        It handles different input formats and routes them to the appropriate internal model.

        Args:
            inputs (dict): Dictionary containing input tensors
            training (bool, optional): Whether the model is in training mode

        Returns:
            tf.Tensor: Model predictions
        """
        # Training mode - use training model
        if training:
            return self._handle_training(inputs)

        # Inference mode - use scorer model
        return self._handle_inference(inputs)

    def _handle_training(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Handle the forward pass during training mode.

        Args:
            inputs (dict): Dictionary containing:
                - 'hist_tokens': User history tokens, shape (batch_size, history_len, title_len)
                - 'user_ids': User IDs for each batch item, shape (batch_size, 1)
                - 'cand_tokens': Candidate news tokens, shape (batch_size, num_candidates, title_len)

        Returns:
            tf.Tensor: Softmax probabilities for each candidate
                Shape: (batch_size, num_candidates)
        """
        history_tokens = inputs["hist_tokens"]
        user_ids = inputs["user_ids"]
        candidate_tokens = inputs["cand_tokens"]
        
        return self.training_model([history_tokens, user_ids, candidate_tokens], training=True)

    def _handle_inference(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Handle the forward pass during inference mode.

        Args:
            inputs (dict): Dictionary containing either:
                - 'history_tokens', 'user_ids', 'single_candidate_tokens' for single candidate scoring
                - 'history_tokens', 'user_ids', 'cand_tokens' for multiple candidate scoring

        Returns:
            tf.Tensor: Model predictions
        """
        # Case 1: Single candidate scoring
        if "single_candidate_tokens" in inputs:
            history_tokens = inputs["history_tokens"]
            user_ids = inputs["user_ids"]
            candidate_tokens = inputs["single_candidate_tokens"]
            return self.scorer_model([history_tokens, user_ids, candidate_tokens], training=False)

        # Case 2: Multiple candidates scoring
        if "cand_tokens" in inputs:
            return self._score_multiple_candidates(inputs)

        raise ValueError(
            "Invalid input format for inference. Expected 'single_candidate_tokens' or 'cand_tokens'"
        )

    def _score_multiple_candidates(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Score multiple candidates for each user in the batch.

        Args:
            inputs (dict): Dictionary containing:
                - 'history_tokens': User history tokens, shape (batch_size, history_len, title_len)
                - 'user_ids': User IDs for each batch item, shape (batch_size, 1)
                - 'cand_tokens': Candidate news tokens, shape (batch_size, num_candidates, title_len)

        Returns:
            tf.Tensor: Scores for all candidates
                Shape: (batch_size, num_candidates)
        """
        history_tokens_batch = inputs["history_tokens"]
        user_ids_batch = inputs["user_ids"]
        candidates_batch = inputs["cand_tokens"]

        batch_size = tf.shape(history_tokens_batch)[0]
        num_candidates = tf.shape(candidates_batch)[1]

        all_scores = []

        # Process each item in the batch
        for i in range(batch_size):
            # Get history and user ID for current item
            current_history_tokens = tf.expand_dims(history_tokens_batch[i], 0)
            current_user_ids = tf.expand_dims(user_ids_batch[i], 0)

            # Score each candidate against this history
            candidate_scores = []
            for j in range(num_candidates):
                current_candidate = tf.expand_dims(candidates_batch[i, j], 0)
                score = self.scorer_model([current_history_tokens, current_user_ids, current_candidate], training=False)
                candidate_scores.append(score)

            # Combine scores for all candidates of this item
            item_scores = tf.concat(candidate_scores, axis=1)
            all_scores.append(item_scores)

        # Combine scores for all items in batch
        return tf.concat(all_scores, axis=0)

    def get_config(self):
        """Returns the configuration of the LSTUR model for serialization.

        Returns:
            dict: Model configuration including all hyperparameters
        """
        config = super().get_config()
        config.update(
            {
                "embedding_size": self.embedding_size,
                "user_embedding_dim": self.user_embedding_dim,
                "cnn_filter_num": self.cnn_filter_num,
                "cnn_kernel_size": self.cnn_kernel_size,
                "attention_hidden_dim": self.attention_hidden_dim,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "max_title_length": self.max_title_length,
                "max_history_length": self.max_history_length,
                "max_impressions_length": self.max_impressions_length,
                "num_users": self.num_users,
                "user_representation_type": self.user_representation_type,
                "process_user_id": self.process_user_id,
                "seed": self.seed,
                "vocab_size": self.vocab_size,
            }
        )
        return config

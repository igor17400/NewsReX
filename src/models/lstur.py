from typing import Tuple, Dict, Any, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .layers import AdditiveAttentionLayer, ComputeMasking, OverwriteMasking
from .base import BaseModel


class NewsInputSplitter(keras.layers.Layer):
    """Splits a concatenated news input into its components (title, category, subcategory)."""

    def __init__(self, title_length: int, **kwargs):
        super().__init__(**kwargs)
        self.title_length = title_length

    def call(self, inputs):
        # Split the concatenated input into its components
        title_tokens = inputs[:, : self.title_length]
        category_id = inputs[:, self.title_length : self.title_length + 1]
        subcategory_id = inputs[:, self.title_length + 1 :]

        return {
            "title_tokens": title_tokens,
            "category_id": category_id,
            "subcategory_id": subcategory_id,
        }

    def compute_output_shape(self, input_shape):
        return {
            "title_tokens": (input_shape[0], self.title_length),
            "category_id": (input_shape[0], 1),
            "subcategory_id": (input_shape[0], 1),
        }


class LSTUR(BaseModel):
    """Neural News Recommendation with Long- and Short-term User Representations (LSTUR) model.

    This model is based on the paper: "Neural News Recommendation with Long- and Short-term User Representations"
    by D. An et al. It captures both long-term and short-term user interests using different neural architectures.

    Key features:
    - Long-term user representation: Uses user ID embeddings to capture stable preferences
    - Short-term user representation: Uses LSTM/GRU to model recent browsing behavior
    - News encoder: Processes news titles using CNN and attention mechanisms
    - Category encoder: Processes category information (as per paper)
    - Subcategory encoder: Processes subcategory information (as per paper)
    - Dual user modeling: Combines long-term and short-term representations
    """

    def __init__(
        self,
        processed_news: Dict[str, Any],
        embedding_size: int = 300,
        cnn_filter_num: int = 400,
        cnn_kernel_size: int = 3,
        attention_hidden_dim: int = 200,
        dropout_rate: float = 0.2,
        cnn_activation: str = "relu",
        max_title_length: int = 30,
        max_history_length: int = 50,
        max_impressions_length: int = 5,
        num_users: int = 100000,  # Total number of users in the dataset
        user_representation_type: str = "lstm",  # "lstm" or "gru"
        user_combination_type: str = "ini",  # "ini" or "con"
        process_user_id: bool = True,  # Only used in base model
        category_embedding_dim: int = 100,
        subcategory_embedding_dim: int = 100,
        use_cat_subcat_encoder: bool = True,
        seed: int = 42,
        name: str = "lstur",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Configurable parameters
        self.embedding_size = embedding_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_kernel_size = cnn_kernel_size
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout_rate = dropout_rate
        self.cnn_activation = cnn_activation
        self.max_title_length = max_title_length
        self.max_history_length = max_history_length
        self.max_impressions_length = max_impressions_length
        self.num_users = num_users
        self.user_representation_type = user_representation_type.lower()
        self.user_combination_type = user_combination_type.lower()
        self.category_embedding_dim = category_embedding_dim
        self.subcategory_embedding_dim = subcategory_embedding_dim
        self.use_cat_subcat_encoder = use_cat_subcat_encoder
        self.seed = seed
        self.process_user_id = process_user_id

        tf.random.set_seed(self.seed)

        # Unpack processed data from the dataset
        self.vocab_size = processed_news["vocab_size"]
        self.embeddings_matrix = processed_news["embeddings"]
        self.num_categories = processed_news.get("num_categories", 0)
        self.num_subcategories = processed_news.get("num_subcategories", 0)

        # Validate user representation type
        if self.user_representation_type not in ["lstm", "gru"]:
            raise ValueError("user_representation_type must be 'lstm' or 'gru'")
        if self.user_combination_type not in ["ini", "con"]:
            raise ValueError("user_combination_type must be 'ini' or 'con'")

        # Build components
        self.word_embedding_layer = self._build_word_embedding_layer()
        self.user_embedding_layer = self._build_user_embedding_layer()

        # Build topic encoders if enabled
        if self.use_cat_subcat_encoder and self.num_categories > 0:
            self.category_encoder = self._build_category_encoder()
        if self.use_cat_subcat_encoder and self.num_subcategories > 0:
            self.subcategory_encoder = self._build_subcategory_encoder()

        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder()

        # Build final training and scoring models
        self.training_model, self.scorer_model = self._build_graph_models()

    def _build_word_embedding_layer(self) -> layers.Embedding:
        """Builds the word embedding layer for news titles."""
        return layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=keras.initializers.Constant(self.embeddings_matrix),
            trainable=True,
            mask_zero=False,
            name="word_embedding",
        )

    def _build_user_embedding_layer(self) -> layers.Embedding:
        """Builds the user embedding layer for long-term user representation."""
        return layers.Embedding(
            input_dim=self.num_users + 1,  # +1 for padding/unknown users
            output_dim=self.cnn_filter_num,
            trainable=True,
            embeddings_initializer="zeros",
            name="user_embedding",
        )

    def _build_category_encoder(self) -> keras.Model:
        """Builds the category encoder which processes category information.

        This method creates a Keras model that embeds topic indices and projects them
        into the same dimension as the CNN filter. The resulting model outputs a dense
        vector representation for each topic.

        Returns:
            keras.Model: A Keras model that encodes topic information.
        """
        input_category = keras.Input(shape=(1,), dtype="int32", name="category_id")

        category_embedding = layers.Embedding(
            self.num_categories + 1,
            self.category_embedding_dim,
            trainable=True,
            name="category_embedding",
        )

        category_emb = category_embedding(input_category)
        category_dense = layers.Dense(
            self.cnn_filter_num,
            activation=self.cnn_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            name="category_projection",
        )(category_emb)
        category_vector = layers.Reshape((1, self.cnn_filter_num))(category_dense)

        return keras.Model(input_category, category_vector, name="category_encoder")

    def _build_subcategory_encoder(self) -> keras.Model:
        """Builds the subcategory encoder which processes subcategory information.

        This method creates a Keras model that embeds subtopic indices and projects them
        into the same dimension as the CNN filter. The resulting model outputs a dense
        vector representation for each subtopic.

        Returns:
            keras.Model: A Keras model that encodes subtopic information.
        """
        input_subcategory = keras.Input(shape=(1,), dtype="int32", name="subcategory_id")

        subcategory_embedding = layers.Embedding(
            self.num_subcategories + 1,
            self.subcategory_embedding_dim,
            trainable=True,
            name="subcategory_embedding",
        )

        subcategory_emb = subcategory_embedding(input_subcategory)
        subcategory_dense = layers.Dense(
            self.cnn_filter_num,
            activation=self.cnn_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            name="subcategory_projection",
        )(subcategory_emb)
        subcategory_vector = layers.Reshape((1, self.cnn_filter_num))(subcategory_dense)

        return keras.Model(input_subcategory, subcategory_vector, name="subcategory_encoder")

    def _build_newsencoder(self) -> keras.Model:
        """Builds the news encoder which processes news titles and topics."""
        if self.use_cat_subcat_encoder and self.num_categories > 0 and self.num_subcategories > 0:
            # News input includes title + category + subcategory
            news_input = keras.Input(
                shape=(self.max_title_length + 2,),  # title + category + subcategory
                dtype="int32",
                name="news_tokens",
            )

            # Split the input into its components
            news_components = NewsInputSplitter(
                title_length=self.max_title_length,
                name="news_input_splitter",
            )(news_input)

            # Title processing
            embedded_title = self.word_embedding_layer(news_components["title_tokens"])
            title_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)(embedded_title)

            title_cnn = layers.Conv1D(
                self.cnn_filter_num,
                self.cnn_kernel_size,
                activation=self.cnn_activation,
                padding="same",
                name="title_cnn",
                bias_initializer=keras.initializers.Zeros(),
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            )(title_dropout)

            title_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)(title_cnn)
            masked_title = layers.Masking()(
                OverwriteMasking()(
                    [title_dropout, ComputeMasking()(news_components["title_tokens"])]
                )
            )

            title_attention = AdditiveAttentionLayer(
                self.attention_hidden_dim, name="title_word_attention"
            )(masked_title)
            title_vector = layers.Reshape((1, self.cnn_filter_num))(title_attention)

            # Category and subcategory processing
            category_vector = self.category_encoder(news_components["category_id"])
            subcategory_vector = self.subcategory_encoder(news_components["subcategory_id"])

            # Concatenate all views: e = [et, ev, esv]
            concat_views = layers.Concatenate(axis=1)(
                [title_vector, category_vector, subcategory_vector]
            )
            news_vector = AdditiveAttentionLayer(self.attention_hidden_dim, name="view_attention")(
                concat_views
            )

            return keras.Model(news_input, news_vector, name="news_encoder")

        # Fallback to title-only encoding (original behavior)
        input_title = keras.Input(shape=(self.max_title_length,), dtype="int32", name="news_tokens")

        embedded_title = self.word_embedding_layer(input_title)
        title_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)(embedded_title)

        title_cnn = layers.Conv1D(
            self.cnn_filter_num,
            self.cnn_kernel_size,
            activation=self.cnn_activation,
            padding="same",
            name="title_cnn",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(title_dropout)

        title_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)(title_cnn)
        masked_title = layers.Masking()(
            OverwriteMasking()([title_dropout, ComputeMasking()(input_title)])
        )

        title_attention = AdditiveAttentionLayer(
            self.attention_hidden_dim, name="title_word_attention"
        )(masked_title)

        return keras.Model(input_title, title_attention, name="news_encoder")

    def _build_userencoder(self) -> keras.Model:
        """Builds the user encoder which combines long-term and short-term representations."""
        if self.use_cat_subcat_encoder and self.num_categories > 0 and self.num_subcategories > 0:
            # Input includes title + category + subcategory concatenated
            history_input = keras.Input(
                shape=(
                    self.max_history_length,
                    self.max_title_length + 2,
                ),  # title + category + subcategory
                dtype="int32",
                name="history_tokens",
            )
        else:
            # Original behavior - just title tokens
            history_input = keras.Input(
                shape=(self.max_history_length, self.max_title_length),
                dtype="int32",
                name="history_tokens",
            )

        user_ids = keras.Input(
            shape=(1,),
            dtype="int32",
            name="user_ids",
        )

        # Use the existing user embedding layer - direct usage like original
        long_u_emb = layers.Reshape((self.cnn_filter_num,))(self.user_embedding_layer(user_ids))

        # Process history through news encoder
        click_title_presents = layers.TimeDistributed(self.newsencoder)(history_input)

        if self.user_combination_type == "ini":
            if self.user_representation_type == "lstm":
                user_present = layers.LSTM(
                    self.cnn_filter_num,
                    kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                    recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                    bias_initializer=keras.initializers.Zeros(),
                )(
                    layers.Masking(mask_value=0.0)(click_title_presents),
                    initial_state=[long_u_emb, long_u_emb],  # LSTM needs [h, c]
                )
            else:  # gru
                user_present = layers.GRU(
                    self.cnn_filter_num,
                    kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                    recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                    bias_initializer=keras.initializers.Zeros(),
                )(
                    layers.Masking(mask_value=0.0)(click_title_presents),
                    initial_state=[long_u_emb],  # GRU only needs [h]
                )
        elif self.user_combination_type == "con":
            if self.user_representation_type == "lstm":
                short_uemb = layers.LSTM(
                    self.cnn_filter_num,
                    kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                    recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                    bias_initializer=keras.initializers.Zeros(),
                )(layers.Masking(mask_value=0.0)(click_title_presents))
            else:  # gru
                short_uemb = layers.GRU(
                    self.cnn_filter_num,
                    kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                    recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                    bias_initializer=keras.initializers.Zeros(),
                )(layers.Masking(mask_value=0.0)(click_title_presents))

            user_present = layers.Concatenate()([short_uemb, long_u_emb])
            user_present = layers.Dense(
                self.cnn_filter_num,
                bias_initializer=keras.initializers.Zeros(),
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            )(user_present)

        return keras.Model([history_input, user_ids], user_present, name="user_encoder")

    def _build_graph_models(self) -> Tuple[keras.Model, keras.Model]:
        """Builds the main training and scoring models."""
        # --- Inputs for Training Model ---
        if self.use_cat_subcat_encoder and self.num_categories > 0 and self.num_subcategories > 0:
            # History inputs with category and subcategory information
            history_tokens_input_train = keras.Input(
                shape=(
                    self.max_history_length,
                    self.max_title_length + 2,
                ),  # title + category + subcategory
                dtype="int32",
                name="history_tokens_train",
            )
            candidate_tokens_input_train = keras.Input(
                shape=(
                    self.max_impressions_length,
                    self.max_title_length + 2,
                ),  # title + topic + subtopic
                dtype="int32",
                name="candidate_tokens_train",
            )
        else:
            # Original behavior - just title tokens
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

        user_ids = keras.Input(
            shape=(1,),  # Single user ID per batch item
            dtype="int32",
            name="user_ids_train",
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

        training_model = keras.Model(
            inputs=[history_tokens_input_train, user_ids, candidate_tokens_input_train],
            outputs=preds_train,
            name="lstur_training_model",
        )

        # --- Inputs for Scorer Model ---
        # History Inputs
        hist_tokens_input_score = keras.Input(
            shape=(self.max_history_length, self.max_title_length),
            dtype="int32",
            name="history_tokens_score",
        )

        if self.use_cat_subcat_encoder and self.num_categories > 0 and self.num_subcategories > 0:
            # History inputs with category and subcategory information
            hist_category_input_score = keras.Input(
                shape=(self.max_history_length, 1),
                dtype="int32",
                name="history_category_score",
            )
            hist_subcategory_input_score = keras.Input(
                shape=(self.max_history_length, 1),
                dtype="int32",
                name="history_subcategory_score",
            )

        # User IDs
        user_ids_score = keras.Input(
            shape=(1,),  # Single user ID per batch item
            dtype="int32",
            name="user_ids_score",
        )

        # Candidate Inputs
        cand_tokens_input_score = keras.Input(
            shape=(1, self.max_title_length),
            dtype="int32",
            name="candidate_tokens_score",
        )

        if self.use_cat_subcat_encoder and self.num_categories > 0 and self.num_subcategories > 0:
            # Candidate inputs with category and subcategory information
            cand_category_input_score = keras.Input(
                shape=(1, 1),
                dtype="int32",
                name="candidate_category_score",
            )
            cand_subcategory_input_score = keras.Input(
                shape=(1, 1),
                dtype="int32",
                name="candidate_subcategory_score",
            )

        if self.use_cat_subcat_encoder and self.num_categories > 0 and self.num_subcategories > 0:
            history_concat_score = layers.Concatenate(axis=-1)(
                [
                    hist_tokens_input_score,
                    hist_category_input_score,
                    hist_subcategory_input_score,
                ]
            )
            candidate_concat_score = layers.Concatenate(axis=-1)(
                [
                    cand_tokens_input_score,
                    cand_category_input_score,
                    cand_subcategory_input_score,
                ]
            )
            # Reshape to remove the extra dimension, similar to NAML model
            candidate_concat_score = layers.Reshape((-1,))(candidate_concat_score)

        # --- Scorer Model Graph ---
        if self.use_cat_subcat_encoder and self.num_categories > 0 and self.num_subcategories > 0:
            user_representation_score = self.userencoder([history_concat_score, user_ids_score])
            single_candidate_representation_score = self.newsencoder(candidate_concat_score)
        else:
            user_representation_score = self.userencoder([hist_tokens_input_score, user_ids_score])
            single_candidate_representation_score = self.newsencoder(cand_tokens_input_score)

        pred_score = layers.Dot(axes=-1, name="dot_product_score")(
            [single_candidate_representation_score, user_representation_score]
        )

        # Apply sigmoid for single prediction probability
        pred_score = layers.Activation("sigmoid", name="sigmoid_activation_score")(pred_score)

        if self.use_cat_subcat_encoder and self.num_categories > 0 and self.num_subcategories > 0:
            scorer_model = keras.Model(
                inputs=[history_concat_score, user_ids_score, candidate_concat_score],
                outputs=pred_score,
                name="lstur_scorer_model",
            )
        else:
            scorer_model = keras.Model(
                inputs=[hist_tokens_input_score, user_ids_score, cand_tokens_input_score],
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
                - 'hist_category': User history categories (if category encoder enabled)
                - 'hist_subcategory': User history subcategories (if category encoder enabled)
                - 'cand_category': Candidate news categories (if category encoder enabled)
                - 'cand_subcategory': Candidate news subcategories (if category encoder enabled)

        Returns:
            tf.Tensor: Softmax probabilities for each candidate
                Shape: (batch_size, num_candidates)
        """
        history_tokens = inputs["hist_tokens"]
        user_ids = inputs["user_ids"]
        candidate_tokens = inputs["cand_tokens"]

        # Handle topic encoder inputs if enabled
        if self.use_cat_subcat_encoder and self.num_categories > 0 and self.num_subcategories > 0:
            # Concatenate history inputs: title + category + subcategory
            hist_category = inputs.get("hist_category")
            hist_subcategory = inputs.get("hist_subcategory")
            if hist_category is not None and hist_subcategory is not None:
                # Expand dimensions to match the rank of title tokens
                hist_category = keras.ops.expand_dims(hist_category, axis=-1)
                hist_subcategory = keras.ops.expand_dims(hist_subcategory, axis=-1)
                history_tokens = keras.ops.concatenate(
                    [history_tokens, hist_category, hist_subcategory], axis=-1
                )

            # Concatenate candidate inputs: title + category + subcategory
            cand_category = inputs.get("cand_category")
            cand_subcategory = inputs.get("cand_subcategory")
            if cand_category is not None and cand_subcategory is not None:
                # Expand dimensions to match the rank of title tokens
                cand_category = keras.ops.expand_dims(cand_category, axis=-1)
                cand_subcategory = keras.ops.expand_dims(cand_subcategory, axis=-1)
                candidate_tokens = keras.ops.concatenate(
                    [candidate_tokens, cand_category, cand_subcategory], axis=-1
                )

        return self.training_model([history_tokens, user_ids, candidate_tokens], training=True)

    def _handle_inference(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Handle the forward pass during inference mode.

        Args:
            inputs (dict): Dictionary containing either:
                - 'history_tokens', 'user_ids', 'single_candidate_tokens' for single candidate scoring
                - 'history_tokens', 'user_ids', 'cand_tokens' for multiple candidate scoring
                - Category/subcategory inputs if category encoder is enabled

        Returns:
            tf.Tensor: Model predictions
        """
        # Case 1: Single candidate scoring
        if "single_candidate_tokens" in inputs:
            history_tokens = inputs["history_tokens"]
            user_ids = inputs["user_ids"]
            candidate_tokens = inputs["single_candidate_tokens"]

            # Handle topic encoder inputs if enabled
            if (
                self.use_cat_subcat_encoder
                and self.num_categories > 0
                and self.num_subcategories > 0
            ):
                # Concatenate history inputs: title + category + subcategory
                hist_category = inputs.get("history_category")
                hist_subcategory = inputs.get("history_subcategory")
                if hist_category is not None and hist_subcategory is not None:
                    # Expand dimensions to match the rank of title tokens
                    hist_category = keras.ops.expand_dims(hist_category, axis=-1)
                    hist_subcategory = keras.ops.expand_dims(hist_subcategory, axis=-1)
                    history_tokens = keras.ops.concatenate(
                        [history_tokens, hist_category, hist_subcategory], axis=-1
                    )

                # Concatenate candidate inputs: title + category + subcategory
                cand_category = inputs.get("single_candidate_category")
                cand_subcategory = inputs.get("single_candidate_subcategory")
                if cand_category is not None and cand_subcategory is not None:
                    # Expand dimensions to match the rank of title tokens
                    cand_category = keras.ops.expand_dims(cand_category, axis=-1)
                    cand_subcategory = keras.ops.expand_dims(cand_subcategory, axis=-1)
                    candidate_tokens = keras.ops.concatenate(
                        [candidate_tokens, cand_category, cand_subcategory], axis=-1
                    )

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
                - Category/subcategory inputs if category encoder is enabled

        Returns:
            tf.Tensor: Scores for all candidates
                Shape: (batch_size, num_candidates)
        """
        history_tokens_batch = inputs["history_tokens"]
        user_ids_batch = inputs["user_ids"]
        candidates_batch = inputs["cand_tokens"]

        # Handle topic encoder inputs if enabled
        if self.use_cat_subcat_encoder and self.num_categories > 0 and self.num_subcategories > 0:
            # Concatenate history inputs: title + category + subcategory
            hist_category = inputs.get("history_category")
            hist_subcategory = inputs.get("history_subcategory")
            if hist_category is not None and hist_subcategory is not None:
                history_tokens_batch = keras.ops.concatenate(
                    [history_tokens_batch, hist_category, hist_subcategory], axis=-1
                )

            # Concatenate candidate inputs: title + category + subcategory
            cand_category = inputs.get("cand_category")
            cand_subcategory = inputs.get("cand_subcategory")
            if cand_category is not None and cand_subcategory is not None:
                candidates_batch = keras.ops.concatenate(
                    [candidates_batch, cand_category, cand_subcategory], axis=-1
                )

        batch_size = tf.shape(history_tokens_batch)[0]
        num_candidates = tf.shape(candidates_batch)[1]

        all_scores = []

        # Process each item in the batch
        for i in range(batch_size):
            # Get history and user ID for current item
            current_history_tokens = keras.ops.expand_dims(history_tokens_batch[i], 0)
            current_user_ids = keras.ops.expand_dims(user_ids_batch[i], 0)

            # Score each candidate against this history
            candidate_scores = []
            for j in range(num_candidates):
                current_candidate = keras.ops.expand_dims(candidates_batch[i, j], 0)
                score = self.scorer_model(
                    [current_history_tokens, current_user_ids, current_candidate], training=False
                )
                candidate_scores.append(score)

            # Combine scores for all candidates of this item
            item_scores = keras.ops.concatenate(candidate_scores, axis=1)
            all_scores.append(item_scores)

        # Combine scores for all items in batch
        return keras.ops.concatenate(all_scores, axis=0)

    def get_config(self):
        """Returns the configuration of the LSTUR model for serialization.

        Returns:
            dict: Model configuration including all hyperparameters
        """
        config = super().get_config()
        config.update(
            {
                "embedding_size": self.embedding_size,
                "cnn_filter_num": self.cnn_filter_num,
                "cnn_kernel_size": self.cnn_kernel_size,
                "attention_hidden_dim": self.attention_hidden_dim,
                "dropout_rate": self.dropout_rate,
                "cnn_activation": self.cnn_activation,
                "max_title_length": self.max_title_length,
                "max_history_length": self.max_history_length,
                "max_impressions_length": self.max_impressions_length,
                "num_users": self.num_users,
                "user_representation_type": self.user_representation_type,
                "user_combination_type": self.user_combination_type,
                "category_embedding_dim": self.category_embedding_dim,
                "subcategory_embedding_dim": self.subcategory_embedding_dim,
                "use_cat_subcat_encoder": self.use_cat_subcat_encoder,
                "seed": self.seed,
                "vocab_size": self.vocab_size,
            }
        )
        return config

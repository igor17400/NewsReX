from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from .layers import AdditiveAttentionLayer
from .base import BaseModel


class NewsInputSplitter(tf.keras.layers.Layer):
    """Splits a concatenated news input into its components (title, abstract, category, subcategory)."""

    def __init__(self, title_length: int, abstract_length: int, **kwargs):
        super().__init__(**kwargs)
        self.title_length = title_length
        self.abstract_length = abstract_length

    def call(self, inputs):
        # Split the concatenated input into its components
        title_tokens = inputs[:, : self.title_length]
        abstract_tokens = inputs[:, self.title_length : self.title_length + self.abstract_length]
        category_id = inputs[
            :,
            self.title_length + self.abstract_length : self.title_length + self.abstract_length + 1,
        ]
        subcategory_id = inputs[:, self.title_length + self.abstract_length + 1 :]

        return {
            "title_tokens": title_tokens,
            "abstract_tokens": abstract_tokens,
            "category_id": category_id,
            "subcategory_id": subcategory_id,
        }

    def compute_output_shape(self, input_shape):
        return {
            "title_tokens": (input_shape[0], self.title_length),
            "abstract_tokens": (input_shape[0], self.abstract_length),
            "category_id": (input_shape[0], 1),
            "subcategory_id": (input_shape[0], 1),
        }


class NAML(BaseModel):
    """Neural Attentive Multi-View Learning (NAML) model for news recommendation.

    This model is based on the paper: "Neural News Recommendation with Attentive Multi-View Learning"
    by C. Wu et al. It learns news representations from multiple views (title, abstract, category)
    and user representations from their browsing history.

    Key features:
    - Multi-view news encoding: Processes title, abstract, and categories separately.
    - View-level attention: Combines different news views into a unified representation.
    - Attentive user encoding: Uses multi-head self-attention over historical news.
    """

    def __init__(
        self,
        processed_news: Dict[str, Any],
        max_title_length: int,
        max_abstract_length: int,
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
        process_user_id: bool = False,  # Only used in base model
        seed: int = 42,
        name: str = "naml",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Configurable parameters
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length
        self.embedding_size = embedding_size
        self.category_embedding_dim = category_embedding_dim
        self.subcategory_embedding_dim = subcategory_embedding_dim
        self.cnn_filter_num = cnn_filter_num
        self.cnn_kernel_size = cnn_kernel_size
        self.word_attention_query_dim = word_attention_query_dim
        self.view_attention_query_dim = view_attention_query_dim
        self.user_attention_query_dim = user_attention_query_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.max_history_length = max_history_length
        self.max_impressions_length = max_impressions_length
        self.process_user_id = process_user_id
        self.seed = seed

        tf.random.set_seed(self.seed)

        # Unpack processed data from the dataset
        self.vocab_size = processed_news["vocab_size"]
        self.num_categories = processed_news["num_categories"]
        self.num_subcategories = processed_news["num_subcategories"]
        self.embeddings_matrix = processed_news["embeddings"]

        # Build reusable embedding layers
        self.word_embedding_layer = self._build_word_embedding_layer()
        self.category_embedding_layer = self._build_category_embedding_layer()
        self.subcategory_embedding_layer = self._build_subcategory_embedding_layer()

        # Build core model components
        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder()

        # Build final training and scoring models
        self.training_model, self.scorer_model = self._build_graph_models()

    def _build_word_embedding_layer(self) -> layers.Embedding:
        return layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.Constant(self.embeddings_matrix),
            trainable=True,
            mask_zero=False,
            name="word_embedding",
        )

    def _build_title_encoder(self) -> tf.keras.Model:
        """Builds the title encoder which processes the title of a news article."""
        input_title = tf.keras.Input(
            shape=(self.max_title_length,), dtype="int32", name="title_tokens"
        )
        embedded_title = self.word_embedding_layer(input_title)

        title_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)(embedded_title)
        title_cnn = layers.Conv1D(
            self.cnn_filter_num,
            self.cnn_kernel_size,
            activation=self.activation,
            padding="same",
            name="title_cnn",
        )(title_dropout)

        title_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)(title_cnn)

        title_attention = AdditiveAttentionLayer(
            self.word_attention_query_dim, name="title_word_attention"
        )(title_dropout)
        title_vector = layers.Reshape((1, self.cnn_filter_num))(title_attention)

        return tf.keras.Model(input_title, title_vector, name="title_encoder")

    def _build_abstract_encoder(self) -> tf.keras.Model:
        """Builds the abstract encoder which processes the abstract of a news article."""
        input_abstract = tf.keras.Input(
            shape=(self.max_abstract_length,), dtype="int32", name="abstract_tokens"
        )

        embedded_abstract = self.word_embedding_layer(input_abstract)
        abstract_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)(embedded_abstract)

        abstract_cnn = layers.Conv1D(
            self.cnn_filter_num,
            self.cnn_kernel_size,
            activation=self.activation,
            padding="same",
            name="abstract_cnn",
        )(abstract_dropout)
        abstract_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)(abstract_cnn)

        abstract_attention = AdditiveAttentionLayer(
            self.word_attention_query_dim, name="abstract_word_attention"
        )(abstract_dropout)
        abstract_vector = layers.Reshape((1, self.cnn_filter_num))(abstract_attention)

        return tf.keras.Model(input_abstract, abstract_vector, name="abstract_encoder")

    def _build_category_embedding_layer(self) -> layers.Embedding:
        """Builds the category embedding and projection layer.

        This method creates a Keras model that embeds category indices and projects them into the same dimension as the CNN filter. The resulting model outputs a dense vector representation for each category.

        Returns:
            layers.Embedding: A Keras model that encodes category information.
        """
        input_category = tf.keras.Input(shape=(1,), dtype="int32", name="category_id")

        category_embedding = layers.Embedding(
            self.num_categories + 1,
            self.category_embedding_dim,
            trainable=True,
            name="category_embedding",
        )

        cat_emb = category_embedding(input_category)
        cat_dense = layers.Dense(
            self.cnn_filter_num,
            activation=self.activation,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
            name="category_projection",
        )(cat_emb)
        cat_vector = layers.Reshape((1, self.cnn_filter_num))(cat_dense)

        return tf.keras.Model(input_category, cat_vector, name="category_encoder")

    def _build_subcategory_embedding_layer(self) -> layers.Embedding:
        """Builds the subcategory embedding and projection layer.

        This method creates a Keras model that embeds subcategory indices and projects them into the same dimension as the CNN filter. The resulting model outputs a dense vector representation for each subcategory.

        Returns:
            layers.Embedding: A Keras model that encodes subcategory information.
        """
        input_subcategory = tf.keras.Input(shape=(1,), dtype="int32", name="subcategory_id")

        subcategory_embedding = layers.Embedding(
            self.num_subcategories + 1,
            self.subcategory_embedding_dim,
            trainable=True,
            name="subcategory_embedding",
        )

        subcat_emb = subcategory_embedding(input_subcategory)
        subcat_dense = layers.Dense(
            self.cnn_filter_num,
            activation=self.activation,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
            name="subcategory_projection",
        )(subcat_emb)
        pred_subcat = layers.Reshape((1, self.cnn_filter_num))(subcat_dense)
        return tf.keras.Model(input_subcategory, pred_subcat, name="subcategory_encoder")

    def _build_newsencoder(self) -> tf.keras.Model:
        """Builds the news encoder which processes multiple views of a news article."""
        # --- Define Inputs for a single news article ---
        # news_input -> is the input that describes a single news article
        # It's composed of the title + abstract + category + subcategory (that's where the 2 comes from)
        news_input = tf.keras.Input(
            shape=(self.max_title_length + self.max_abstract_length + 2,),
            dtype="int32",
            name="news_tokens",
        )

        # Split the input into its components
        news_components = NewsInputSplitter(
            title_length=self.max_title_length,
            abstract_length=self.max_abstract_length,
            name="news_input_splitter",
        )(news_input)

        # --- Title Encoder ---
        title_encoder = self._build_title_encoder()
        title_vector = title_encoder(news_components["title_tokens"])

        # --- Abstract Encoder ---
        abstract_encoder = self._build_abstract_encoder()
        abstract_vector = abstract_encoder(news_components["abstract_tokens"])

        # --- Category Encoder ---
        category_encoder = self._build_category_embedding_layer()
        category_vector = category_encoder(news_components["category_id"])

        # --- Subcategory Encoder ---
        subcategory_encoder = self._build_subcategory_embedding_layer()
        subcategory_vector = subcategory_encoder(news_components["subcategory_id"])

        # --- Combine all views about a news article ---
        concat_views = layers.Concatenate(axis=1)(
            [title_vector, abstract_vector, category_vector, subcategory_vector]
        )
        news_vector = AdditiveAttentionLayer(self.view_attention_query_dim, name="view_attention")(
            concat_views
        )

        return tf.keras.Model(news_input, news_vector, name="news_encoder")

    def _build_userencoder(self) -> tf.keras.Model:
        """Builds the user encoder which processes the user's news browsing history."""
        # --- Define Inputs for a user's history ---
        # his_input -> is the input for the news history from the user
        # It's composed of the title + abstract + category + subcategory (that's where the 2 comes from)
        hist_input = tf.keras.Input(
            shape=(self.max_history_length, self.max_title_length + self.max_abstract_length + 2),
            dtype="int32",
            name="hist_tokens",
        )

        news_vectors = layers.TimeDistributed(self.newsencoder, name="td_history_news_encoder")(
            hist_input
        )

        user_representation = AdditiveAttentionLayer(
            self.user_attention_query_dim, name="user_additive_attention"
        )(news_vectors)

        return tf.keras.Model(hist_input, user_representation, name="user_encoder")

    def _build_graph_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Builds the main training and scoring models."""
        # --- Inputs for Training Model ---

        # History Inputs
        hist_title_tokens = tf.keras.Input(
            shape=(self.max_history_length, self.max_title_length), dtype="int32"
        )
        hist_abstract_tokens = tf.keras.Input(
            shape=(self.max_history_length, self.max_abstract_length), dtype="int32"
        )
        hist_category = tf.keras.Input(shape=(self.max_history_length, 1), dtype="int32")
        hist_subcategory = tf.keras.Input(shape=(self.max_history_length, 1), dtype="int32")

        # Candidate Inputs
        cand_title_tokens = tf.keras.Input(
            shape=(self.max_impressions_length, self.max_title_length), dtype="int32"
        )
        cand_abstract_tokens = tf.keras.Input(
            shape=(self.max_impressions_length, self.max_abstract_length), dtype="int32"
        )
        cand_category = tf.keras.Input(shape=(self.max_impressions_length, 1), dtype="int32")
        cand_subcategory = tf.keras.Input(shape=(self.max_impressions_length, 1), dtype="int32")

        # Concatenate inputs
        history_concat = layers.Concatenate(axis=-1)(
            [
                hist_title_tokens,
                hist_abstract_tokens,
                hist_category,
                hist_subcategory,
            ]
        )
        candidate_concat = layers.Concatenate(axis=-1)(
            [
                cand_title_tokens,
                cand_abstract_tokens,
                cand_category,
                cand_subcategory,
            ]
        )

        # --- Training Model Graph ---
        user_representation_train = self.userencoder(history_concat)
        candidate_news_representation_train = layers.TimeDistributed(
            self.newsencoder, name="td_candidate_news_encoder_train"
        )(candidate_concat)

        scores = layers.Dot(axes=-1, name="dot_product_train")(
            [candidate_news_representation_train, user_representation_train]
        )
        preds_train = layers.Activation("softmax", name="softmax_activation_train")(scores)

        training_model = tf.keras.Model(
            inputs=[history_concat, candidate_concat],
            outputs=preds_train,
            name="naml_training_model",
        )

        # --- Inputs for Scorer Model ---
        # History Inputs
        hist_tokens_input_score = tf.keras.Input(
            shape=(self.max_history_length, self.max_title_length), dtype="int32"
        )
        hist_abstract_tokens_input_score = tf.keras.Input(
            shape=(self.max_history_length, self.max_abstract_length), dtype="int32"
        )
        hist_category_input_score = tf.keras.Input(
            shape=(self.max_history_length, 1), dtype="int32"
        )
        hist_subcategory_input_score = tf.keras.Input(
            shape=(self.max_history_length, 1), dtype="int32"
        )

        # Candidate Inputs
        cand_tokens_input_score = tf.keras.Input(shape=(1, self.max_title_length), dtype="int32")
        cand_abstract_tokens_input_score = tf.keras.Input(
            shape=(1, self.max_abstract_length), dtype="int32"
        )
        cand_category_input_score = tf.keras.Input(shape=(1, 1), dtype="int32")
        cand_subcategory_input_score = tf.keras.Input(shape=(1, 1), dtype="int32")

        history_concat_score = layers.Concatenate(axis=-1)(
            [
                hist_tokens_input_score,
                hist_abstract_tokens_input_score,
                hist_category_input_score,
                hist_subcategory_input_score,
            ]
        )
        candidate_concat_score = layers.Concatenate(axis=-1)(
            [
                cand_tokens_input_score,
                cand_abstract_tokens_input_score,
                cand_category_input_score,
                cand_subcategory_input_score,
            ]
        )
        candidate_concat_score = layers.Reshape((-1,))(candidate_concat_score)

        # --- Scorer Model Graph ---
        user_representation_score = self.userencoder(history_concat_score)
        single_candidate_representation_score = self.newsencoder(candidate_concat_score)

        pred_score = layers.Dot(axes=-1, name="dot_product_score")(
            [single_candidate_representation_score, user_representation_score]
        )
        pred_score = layers.Activation("sigmoid", name="sigmoid_activation_score")(pred_score)

        scorer_model = tf.keras.Model(
            inputs=[history_concat_score, candidate_concat_score],
            outputs=pred_score,
            name="naml_scorer_model",
        )

        return training_model, scorer_model

    def call(self, inputs: Dict[str, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        # Training mode - use training model
        if training:
            return self._handle_training(inputs)

        # Inference mode - use scorer model
        return self._handle_inference(inputs)

    def _handle_training(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Handle the forward pass during training mode.

        This method processes inputs for the training model, which expects a list of two tensors:
        [history_tokens, candidate_tokens]. The training model outputs probabilities for each
        candidate in the batch.

        Args:
            inputs (dict): Dictionary containing:
                - 'hist_tokens': User history tokens, shape (batch_size, history_len, title_len)
                - 'cand_tokens': Candidate news tokens, shape (batch_size, num_candidates, title_len)
                - 'hist_abstract_tokens': User history abstract tokens
                - 'cand_abstract_tokens': Candidate news abstract tokens
                - 'hist_category': User history categories
                - 'cand_category': Candidate news categories
                - 'hist_subcategory': User history subcategories
                - 'cand_subcategory': Candidate news subcategories

        Returns:
            tf.Tensor: Softmax probabilities for each candidate
                Shape: (batch_size, num_candidates)
        """
        # Get input tensors
        hist_title_tokens = inputs["hist_tokens"]
        hist_abstract_tokens = inputs["hist_abstract_tokens"]
        hist_category = inputs["hist_category"]
        hist_subcategory = inputs["hist_subcategory"]
        cand_title_tokens = inputs["cand_tokens"]
        cand_abstract_tokens = inputs["cand_abstract_tokens"]
        cand_category = inputs["cand_category"]
        cand_subcategory = inputs["cand_subcategory"]

        # Reshape category and subcategory tensors to match the expected input shape
        hist_category = tf.expand_dims(hist_category, axis=-1)
        hist_subcategory = tf.expand_dims(hist_subcategory, axis=-1)
        cand_category = tf.expand_dims(cand_category, axis=-1)
        cand_subcategory = tf.expand_dims(cand_subcategory, axis=-1)

        # Concatenate features for history
        hist_features = tf.concat(
            [hist_title_tokens, hist_abstract_tokens, hist_category, hist_subcategory], axis=-1
        )

        # Concatenate features for candidates
        cand_features = tf.concat(
            [cand_title_tokens, cand_abstract_tokens, cand_category, cand_subcategory], axis=-1
        )

        return self.training_model(
            [hist_features, cand_features],
            training=True,
        )

    def _handle_inference(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        if "candidate_news_tokens" in inputs:
            # Check if it's a single candidate or multiple
            candidate_shape = tf.shape(inputs["candidate_news_tokens"])
            if len(candidate_shape) == 2:  # Single candidate: (batch_size, title_len)
                return self.scorer_model(inputs, training=False)
            else:  # Multiple candidates
                return self._score_multiple_candidates(inputs)

        raise ValueError(
            "Invalid input format for NAML inference. Expected 'candidate_news_tokens'"
        )

    def _score_multiple_candidates(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Scores multiple candidates for each history in the batch by iterating."""
        history_batch = {k: v for k, v in inputs.items() if k.startswith("history_")}
        candidates_batch = {k: v for k, v in inputs.items() if k.startswith("candidate_")}

        batch_size = tf.shape(next(iter(history_batch.values())))[0]
        num_candidates = tf.shape(next(iter(candidates_batch.values())))[1]

        all_scores = []
        for i in tf.range(batch_size):
            current_history = {k: tf.expand_dims(v[i], 0) for k, v in history_batch.items()}

            candidate_scores = []
            for j in tf.range(num_candidates):
                current_candidate = {k: v[i, j] for k, v in candidates_batch.items()}
                score = self.scorer_model({**current_history, **current_candidate})
                candidate_scores.append(score)

            item_scores = tf.concat(candidate_scores, axis=1)
            all_scores.append(item_scores)

        return tf.concat(all_scores, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_title_length": self.max_title_length,
                "max_abstract_length": self.max_abstract_length,
                "embedding_size": self.embedding_size,
                "category_embedding_dim": self.category_embedding_dim,
                "subcategory_embedding_dim": self.subcategory_embedding_dim,
                "cnn_filter_num": self.cnn_filter_num,
                "cnn_kernel_size": self.cnn_kernel_size,
                "word_attention_query_dim": self.word_attention_query_dim,
                "view_attention_query_dim": self.view_attention_query_dim,
                "user_attention_query_dim": self.user_attention_query_dim,
                "dropout_rate": self.dropout_rate,
                "seed": self.seed,
                "vocab_size": self.vocab_size,
                "num_categories": self.num_categories,
                "num_subcategories": self.num_subcategories,
            }
        )
        return config

from typing import Any, Optional, Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from .base import BaseNewsRecommender
from .layers import AdditiveSelfAttention


class NewsEncoder(layers.Layer):
    """Encodes news articles into fixed-length vector representations using multi-view learning.

    The encoder processes multiple views of news content:
    1. Title text
    2. Abstract text
    3. Category information
    4. Subcategory information

    Each view is processed through:
    1. Word embeddings (for text) or category embeddings
    2. CNN for feature extraction (for text)
    3. Attention mechanism for each view
    4. View-level attention to combine all views

    Args:
        word_embedding_layer: Layer for word embeddings
        category_embedding_dim: Dimension of category embeddings
        num_categories: Number of unique categories
        num_subcategories: Number of unique subcategories
        cnn_filter_num: Number of CNN filters
        cnn_kernel_size: Size of CNN kernel
        word_attention_query_dim: Dimension of word attention query
        view_attention_query_dim: Dimension of view attention query
        news_encoder_dense_units: Number of units in dense layers
        dropout_rate: Dropout rate
        news_embedding_dim: Final news embedding dimension
    """

    def __init__(
        self,
        word_embedding_layer: layers.Embedding,
        category_embedding_dim: int,
        sub_category_embedding_dim: int,
        num_categories: int,
        num_subcategories: int,
        cnn_filter_num: int,
        cnn_kernel_size: int,
        word_attention_query_dim: int,
        view_attention_query_dim: int,
        news_encoder_dense_units: int,
        dropout_rate: float,
        news_embedding_dim: int,
        name: str = "naml_news_encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.word_embedding_layer = word_embedding_layer

        # CNN layers for text processing
        self.title_cnn = layers.Conv1D(
            filters=cnn_filter_num,
            kernel_size=cnn_kernel_size,
            padding="same",
            activation="relu",
            name="title_cnn",
        )
        self.abstract_cnn = layers.Conv1D(
            filters=cnn_filter_num,
            kernel_size=cnn_kernel_size,
            padding="same",
            activation="relu",
            name="abstract_cnn",
        )

        # Word-level attention for text
        self.title_word_attention = AdditiveSelfAttention(
            query_vector_dim=word_attention_query_dim,
            dropout=dropout_rate,
            name="title_word_attention",
        )
        self.abstract_word_attention = AdditiveSelfAttention(
            query_vector_dim=word_attention_query_dim,
            dropout=dropout_rate,
            name="abstract_word_attention",
        )

        # Category and subcategory embeddings
        self.category_embedding_layer = layers.Embedding(
            input_dim=num_categories + 1,  # +1 for padding/unknown
            output_dim=category_embedding_dim,
            name="category_embedding",
        )
        self.subcategory_embedding_layer = layers.Embedding(
            input_dim=num_subcategories + 1,  # +1 for padding/unknown
            output_dim=sub_category_embedding_dim,
            name="subcategory_embedding",
        )

        # Projection layers for categories
        self.category_projection = layers.Dense(
            cnn_filter_num, activation="relu", name="category_projection"
        )
        self.subcategory_projection = layers.Dense(
            cnn_filter_num, activation="relu", name="subcategory_projection"
        )

        # View-level attention
        self.view_attention = AdditiveSelfAttention(
            query_vector_dim=view_attention_query_dim, dropout=dropout_rate, name="view_attention"
        )

        # Final dense layers
        self.final_dense1 = layers.Dense(
            news_encoder_dense_units, activation="relu", name="news_enc_dense1"
        )
        self.dropout_layer = layers.Dropout(dropout_rate, name="news_enc_dropout")
        self.final_dense2 = layers.Dense(news_embedding_dim, name="news_enc_dense2")

    def call(self, inputs: Dict[str, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """Process news article inputs through the multi-view encoder.

        Args:
            inputs: Dictionary containing:
                - title_tokens: Tokenized title
                - abstract_tokens: Tokenized abstract
                - category_id: Category index
                - subcategory_id: Subcategory index
            training: Whether the layer is being called in training mode

        Returns:
            News article representation vector
        """
        # Process title
        title_tokens = inputs["title_tokens"]
        title_embedded = self.word_embedding_layer(title_tokens)
        title_conv = self.title_cnn(title_embedded)
        title_vector = self.title_word_attention(title_conv, training=training)

        # Process abstract
        abstract_tokens = inputs["abstract_tokens"]
        abstract_embedded = self.word_embedding_layer(abstract_tokens)
        abstract_conv = self.abstract_cnn(abstract_embedded)
        abstract_vector = self.abstract_word_attention(abstract_conv, training=training)

        # Process category
        category_ids = inputs["category_id"]
        if len(category_ids.shape) == 2 and category_ids.shape[1] == 1:
            category_ids = tf.squeeze(category_ids, axis=1)
        category_embedded = self.category_embedding_layer(category_ids)
        category_vector = self.category_projection(category_embedded)

        # Process subcategory
        subcategory_ids = inputs["subcategory_id"]
        if len(subcategory_ids.shape) == 2 and subcategory_ids.shape[1] == 1:
            subcategory_ids = tf.squeeze(subcategory_ids, axis=1)
        subcategory_embedded = self.subcategory_embedding_layer(subcategory_ids)
        subcategory_vector = self.subcategory_projection(subcategory_embedded)

        # Combine all views
        stacked_views = tf.stack(
            [title_vector, abstract_vector, category_vector, subcategory_vector], axis=1
        )
        attended_views = self.view_attention(stacked_views, training=training)

        # Final processing
        dense1_output = self.final_dense1(attended_views)
        dropout_output = self.dropout_layer(dense1_output, training=training)

        return self.final_dense2(dropout_output)


class UserEncoder(layers.Layer):
    """Encodes user's news browsing history into a single vector representation.

    The encoder processes a sequence of news vectors (from NewsEncoder) representing
    the user's click history through:
    1. Multi-head self-attention to capture contextual relationships between news articles
    2. Additive attention to select the most relevant news articles for the final user representation

    Args:
        user_num_attention_heads: Number of attention heads
        user_attention_query_dim: Dimension of attention query vector
        dropout_rate: Dropout rate
        news_embedding_dim: Dimension of input news vectors
    """

    def __init__(
        self,
        user_num_attention_heads: int,
        user_attention_query_dim: int,
        dropout_rate: float,
        news_embedding_dim: int,
        name: str = "naml_user_encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.news_embedding_dim = news_embedding_dim
        self.user_num_attention_heads = user_num_attention_heads

        # Validate dimensions
        if news_embedding_dim % user_num_attention_heads != 0:
            raise ValueError(
                f"news_embedding_dim ({news_embedding_dim}) must be divisible by "
                f"user_num_attention_heads ({user_num_attention_heads}) for MultiHeadAttention."
            )
        head_size = news_embedding_dim // user_num_attention_heads

        # Multi-head self-attention for news sequence
        self.multihead_attention = layers.MultiHeadAttention(
            num_heads=user_num_attention_heads,
            key_dim=head_size,
            dropout=dropout_rate,
            name="user_multihead_attention",
        )

        # Additive attention for final user representation
        self.additive_attention = AdditiveSelfAttention(
            query_vector_dim=user_attention_query_dim,
            dropout=dropout_rate,
            name="user_additive_attention",
        )

    def call(self, news_vectors: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Process sequence of news vectors to create user representation.

        Args:
            news_vectors: Tensor of shape (batch_size, history_length, news_embedding_dim)
                containing news article vectors from user's history
            training: Whether the layer is being called in training mode

        Returns:
            User representation vector
        """
        # Apply multi-head self-attention to capture contextual relationships
        multihead_attended_news = self.multihead_attention(
            query=news_vectors, value=news_vectors, key=news_vectors, training=training
        )

        # Apply additive attention to get final user representation
        return self.additive_attention(multihead_attended_news, training=training)


class ClickPredictor(layers.Layer):
    """Predicts click probability based on user and news vectors.

    The predictor computes the dot product between user and news vectors,
    followed by a softmax activation to get click probabilities.

    Args:
        name: Layer name
    """

    def __init__(self, name: str = "click_predictor", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def call(
        self,
        user_vector: tf.Tensor,
        candidate_news_vector: tf.Tensor,
    ) -> tf.Tensor:
        """Compute click probability for each candidate news article.

        Args:
            user_vector: Tensor of shape (batch_size, embedding_dim)
                containing user representation
            candidate_news_vector: Tensor of shape (batch_size, num_candidates, embedding_dim)
                containing candidate news article representations

        Returns:
            Click probability for each candidate news article
        """
        # Compute dot product between user vector and each candidate news vector
        scores = tf.matmul(candidate_news_vector, tf.expand_dims(user_vector, axis=-1))
        scores = tf.squeeze(scores, axis=-1)

        # Apply softmax to get click probabilities
        return tf.nn.softmax(scores, axis=-1)


class NAML(BaseNewsRecommender):
    """Neural Attentive Multi-View Learning (NAML) model for news recommendation.

    NAML processes news articles through multiple views (title, abstract, category, subcategory)
    and uses attention mechanisms to learn user preferences. The model consists of:
    1. NewsEncoder: Processes each view of a news article and combines them with attention
    2. UserEncoder: Processes user's click history using multi-head attention
    3. Click prediction using dot product and softmax/sigmoid activation

    Args:
        word_embedding_dim: Dimension of word embeddings
        category_embedding_dim: Dimension of category embeddings
        subcategory_embedding_dim: Dimension of subcategory embeddings
        num_filters: Number of CNN filters
        window_sizes: List of CNN window sizes
        news_num_attention_heads: Number of attention heads for news encoder
        news_attention_query_dim: Dimension of attention query vector for news encoder
        user_num_attention_heads: Number of attention heads for user encoder
        user_attention_query_dim: Dimension of attention query vector for user encoder
        dropout_rate: Dropout rate
        name: Model name
    """

    def __init__(
        self,
        word_embedding_dim: int,
        category_embedding_dim: int,
        sub_category_embedding_dim: int,
        num_categories: int,
        num_subcategories: int,
        num_filters: int,
        window_sizes: List[int],
        news_attention_query_dim: int,
        user_num_attention_heads: int,
        user_attention_query_dim: int,
        dropout_rate: float,
        name: str = "naml",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        # News encoder processes each view and combines them
        self.news_encoder = NewsEncoder(
            word_embedding_dim=word_embedding_dim,
            category_embedding_dim=category_embedding_dim,
            sub_category_embedding_dim=sub_category_embedding_dim,
            num_categories=num_categories,
            num_subcategories=num_subcategories,
            cnn_filter_num=num_filters,
            cnn_kernel_size=window_sizes[0],
            word_attention_query_dim=news_attention_query_dim,
            view_attention_query_dim=news_attention_query_dim,
            news_encoder_dense_units=128,
            dropout_rate=dropout_rate,
            news_embedding_dim=word_embedding_dim,
        )

        # User encoder processes click history
        self.user_encoder = UserEncoder(
            user_num_attention_heads=user_num_attention_heads,
            user_attention_query_dim=user_attention_query_dim,
            dropout_rate=dropout_rate,
            news_embedding_dim=word_embedding_dim,  # News vectors have same dim as word embeddings
        )

        # Build both models
        self.model, self.scorer = self._build_model()

    def _build_model(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Build both the training model and the scorer model."""
        # Create input layers for history
        history_title_input = tf.keras.Input(
            shape=(None, None),  # (history_length, title_length)
            dtype="int32",
            name="history_title_input",
        )
        history_abstract_input = tf.keras.Input(
            shape=(None, None),  # (history_length, abstract_length)
            dtype="int32",
            name="history_abstract_input",
        )
        history_category_input = tf.keras.Input(
            shape=(None,),  # (history_length,)
            dtype="int32",
            name="history_category_input",
        )
        history_subcategory_input = tf.keras.Input(
            shape=(None,),  # (history_length,)
            dtype="int32",
            name="history_subcategory_input",
        )

        # Input for training (multiple candidates)
        candidate_title_input = tf.keras.Input(
            shape=(None, None),  # (num_candidates, title_length)
            dtype="int32",
            name="candidate_title_input",
        )
        candidate_abstract_input = tf.keras.Input(
            shape=(None, None),  # (num_candidates, abstract_length)
            dtype="int32",
            name="candidate_abstract_input",
        )
        candidate_category_input = tf.keras.Input(
            shape=(None,),  # (num_candidates,)
            dtype="int32",
            name="candidate_category_input",
        )
        candidate_subcategory_input = tf.keras.Input(
            shape=(None,),  # (num_candidates,)
            dtype="int32",
            name="candidate_subcategory_input",
        )

        # Input for scoring (single candidate)
        candidate_one_title_input = tf.keras.Input(
            shape=(1, None),  # (1, title_length)
            dtype="int32",
            name="candidate_one_title_input",
        )
        candidate_one_abstract_input = tf.keras.Input(
            shape=(1, None),  # (1, abstract_length)
            dtype="int32",
            name="candidate_one_abstract_input",
        )
        candidate_one_category_input = tf.keras.Input(
            shape=(1,),  # (1,)
            dtype="int32",
            name="candidate_one_category_input",
        )
        candidate_one_subcategory_input = tf.keras.Input(
            shape=(1,),  # (1,)
            dtype="int32",
            name="candidate_one_subcategory_input",
        )

        # Process user history
        history_news_vectors = tf.keras.layers.TimeDistributed(self.news_encoder)(
            {
                "title_tokens": history_title_input,
                "abstract_tokens": history_abstract_input,
                "category_id": history_category_input,
                "subcategory_id": history_subcategory_input,
            }
        )
        user_vector = self.user_encoder(history_news_vectors)

        # Training model: process multiple candidates
        candidate_news_vectors = tf.keras.layers.TimeDistributed(self.news_encoder)(
            {
                "title_tokens": candidate_title_input,
                "abstract_tokens": candidate_abstract_input,
                "category_id": candidate_category_input,
                "subcategory_id": candidate_subcategory_input,
            }
        )

        # Use dot product and softmax for training multiple candidates
        dot_product = tf.keras.layers.Dot(axes=-1)([candidate_news_vectors, user_vector])
        pred_scores = tf.keras.layers.Activation("softmax")(dot_product)

        # Scorer model: process single candidate
        candidate_one_news_vector = self.news_encoder(
            {
                "title_tokens": tf.squeeze(candidate_one_title_input, axis=1),
                "abstract_tokens": tf.squeeze(candidate_one_abstract_input, axis=1),
                "category_id": tf.squeeze(candidate_one_category_input, axis=1),
                "subcategory_id": tf.squeeze(candidate_one_subcategory_input, axis=1),
            }
        )

        # Use dot product and sigmoid for scoring single candidate
        dot_product_one = tf.keras.layers.Dot(axes=-1)([candidate_one_news_vector, user_vector])
        pred_one_score = tf.keras.layers.Activation("sigmoid")(dot_product_one)

        # Create models
        model = tf.keras.Model(
            inputs=[
                history_title_input,
                history_abstract_input,
                history_category_input,
                history_subcategory_input,
                candidate_title_input,
                candidate_abstract_input,
                candidate_category_input,
                candidate_subcategory_input,
            ],
            outputs=pred_scores,
            name="naml_model",
        )

        scorer = tf.keras.Model(
            inputs=[
                history_title_input,
                history_abstract_input,
                history_category_input,
                history_subcategory_input,
                candidate_one_title_input,
                candidate_one_abstract_input,
                candidate_one_category_input,
                candidate_one_subcategory_input,
            ],
            outputs=pred_one_score,
            name="naml_scorer",
        )

        return model, scorer

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = True) -> tf.Tensor:
        """Forward pass using the appropriate model."""
        # Adapt input keys to match model expectations
        adapted_inputs = {
            "history_title_input": inputs["history_news_title"],
            "history_abstract_input": inputs["history_news_abstract"],
            "history_category_input": inputs["history_news_category"],
            "history_subcategory_input": inputs["history_news_subcategory"],
        } | (
            {
                "candidate_title_input": inputs["candidate_news_title"],
                "candidate_abstract_input": inputs["candidate_news_abstract"],
                "candidate_category_input": inputs["candidate_news_category"],
                "candidate_subcategory_input": inputs["candidate_news_subcategory"],
            }
            if training
            else {}
        )

        if training:
            return self.model(adapted_inputs)

        # For validation/testing, reshape inputs for scoring
        adapted_inputs = self._reshape_for_scoring(inputs)
        scores = self.scorer(adapted_inputs)

        # Reshape scores to match expected output shape
        num_candidates = inputs["candidate_news_title"].shape[1]
        return self._reshape_scores(scores, num_candidates)

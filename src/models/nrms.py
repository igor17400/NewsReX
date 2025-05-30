from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Dropout,
    Layer,
    MultiHeadAttention,
    TimeDistributed,
)
from .layers import AdditiveSelfAttention
from .base import BaseNewsRecommender

# from models.base import BaseNewsRecommender #TODO: add an abstract base class


class NewsEncoder(Layer):
    """
    Encodes news titles into fixed-length vector representations.

    The encoder processes sequences of word embeddings through:
    1. Multi-head self-attention to capture contextual relationships between words
    2. Additive attention to select the most informative words

    Args:
        multiheads: Number of attention heads
        head_dim: Dimension of each attention head
        attention_hidden_dim: Dimension of attention query vector
        dropout_rate: Dropout rate
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        multiheads: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout_rate: float = 0.2,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout_rate = dropout_rate
        self.seed = seed

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer."""
        self.multihead_attention = MultiHeadAttention(
            num_heads=self.multiheads,
            key_dim=self.head_dim,
            seed=self.seed,
            name="news_multihead_attention",
        )
        self.dropout = Dropout(
            self.dropout_rate,
            seed=self.seed,
            name="news_dropout",
        )
        self.additive_attention = AdditiveSelfAttention(
            query_vector_dim=self.attention_hidden_dim,
            dropout=self.dropout_rate,
            name="news_additive_attention",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Process title embeddings."""
        # Apply dropout
        title_repr = self.dropout(inputs, training=training)

        # Multi-head self attention
        title_repr = self.multihead_attention(
            query=title_repr,
            key=title_repr,
            value=title_repr,
            training=training,
        )

        # Apply dropout after attention
        title_repr = self.dropout(title_repr, training=training)

        # Additive attention
        return self.additive_attention(title_repr, training=training)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor (batch_size, seq_length, embedding_dim)

        Returns:
            Shape of the output tensor (batch_size, embedding_dim)
        """
        return tf.TensorShape([input_shape[0], self.head_dim * self.multiheads])


class UserEncoder(Layer):
    """
    Encodes a user's history of browsed news into a single vector representation.

    The encoder processes a sequence of news vectors through:
    1. Multi-head self-attention to capture contextual relationships between news articles
    2. Additive attention to select the most relevant news articles

    Args:
        multiheads: Number of attention heads
        head_dim: Dimension of each attention head
        attention_hidden_dim: Dimension of attention query vector
        seed: Random seed for reproducibility
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        multiheads: int,
        head_dim: int,
        attention_hidden_dim: int,
        seed: int = 0,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.seed = seed
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer."""
        self.multihead_attention = MultiHeadAttention(
            num_heads=self.multiheads,
            key_dim=self.head_dim,
            seed=self.seed,
            name="user_multihead_attention",
        )
        self.additive_attention = AdditiveSelfAttention(
            query_vector_dim=self.attention_hidden_dim,
            dropout=self.dropout_rate,
            name="user_additive_attention",
        )
        super().build(input_shape)

    def call(self, news_vecs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Process news vectors."""
        # Multi-head self attention
        click_repr = self.multihead_attention(
            query=news_vecs,
            key=news_vecs,
            value=news_vecs,
            training=training,
        )

        # Additive attention for final user representation
        return self.additive_attention(click_repr, training=training)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor (batch_size, seq_length, embedding_dim)

        Returns:
            Shape of the output tensor (batch_size, embedding_dim)
        """
        return tf.TensorShape([input_shape[0], self.head_dim * self.multiheads])


class NRMS(BaseNewsRecommender):
    """Neural News Recommendation with Multi-Head Self-Attention (NRMS) model.

    NRMS processes news articles through:
    1. NewsEncoder: Processes news titles using multi-head attention
    2. UserEncoder: Processes user's click history using multi-head attention
    3. Click prediction using dot product and softmax/sigmoid activation

    Args:
        processed_news: Dictionary containing processed news data
        embedding_size: Dimension of word embeddings
        multiheads: Number of attention heads
        head_dim: Dimension of each attention head
        attention_hidden_dim: Dimension of attention query vector
        dropout_rate: Dropout rate
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        processed_news: Dict[str, Any],
        embedding_size: int = 300,
        multiheads: int = 16,
        head_dim: int = 64,
        attention_hidden_dim: int = 200,
        dropout_rate: float = 0.2,
        seed: int = 42,
        name: str = "nrms",
        loss: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> None:
        if not loss:
            raise ValueError(
                "Loss configuration is required. Please specify loss parameters in the config file."
            )

        loss_name = loss["name"]  # Required field
        loss_kwargs = {k: v for k, v in loss.items() if k != "name"}

        super().__init__(name=name, loss=loss_name, loss_kwargs=loss_kwargs, **kwargs)
        self.embedding_size = embedding_size
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.vocab_size = processed_news["vocab_size"]
        self.embeddings = processed_news["embeddings"]

        # Build both models
        self.model, self.scorer = self._build_model()

    def _build_model(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Build both the training model and the scorer model."""
        # Create input layers
        history_input = tf.keras.Input(
            shape=(None, None),  # (history_length, title_length)
            dtype="int32",
            name="history_input",
        )

        # Input for training (multiple candidates)
        candidate_input = tf.keras.Input(
            shape=(None, None),  # (num_candidates, title_length)
            dtype="int32",
            name="candidate_input",
        )

        # Input for scoring (single candidate)
        candidate_one_input = tf.keras.Input(
            shape=(1, None),  # (1, title_length)
            dtype="int32",
            name="candidate_input_one_instance",
        )

        # Create embedding layer
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.Constant(self.embeddings),
            trainable=True,
            mask_zero=False,
            name="embedding",
        )

        # Create encoders
        news_encoder = NewsEncoder(
            multiheads=self.multiheads,
            head_dim=self.head_dim,
            attention_hidden_dim=self.attention_hidden_dim,
            dropout_rate=self.dropout_rate,
            seed=self.seed,
            name="news_encoder",
        )

        user_encoder = UserEncoder(
            multiheads=self.multiheads,
            head_dim=self.head_dim,
            attention_hidden_dim=self.attention_hidden_dim,
            seed=self.seed,
            name="user_encoder",
        )

        # Process user history
        user_embeds = embedding_layer(history_input)
        user_news_vec = TimeDistributed(news_encoder)(user_embeds)
        user_vec = user_encoder(user_news_vec)

        # Training model: process multiple candidates
        candidate_embeds = embedding_layer(candidate_input)
        candidate_news_vec = TimeDistributed(news_encoder)(candidate_embeds)

        # Use dot product and softmax for training multiple candidates
        dot_product = tf.keras.layers.Dot(axes=-1)([candidate_news_vec, user_vec])
        pred_scores = tf.keras.layers.Activation("softmax")(dot_product)

        # Scorer model: process single candidate
        # Reshape single candidate to remove batch dimension
        candidate_one_reshape = tf.keras.layers.Reshape((-1,))(candidate_one_input)
        candidate_one_embeds = embedding_layer(candidate_one_reshape)
        candidate_one_vec = news_encoder(candidate_one_embeds)

        # Use dot product and sigmoid for scoring single candidate
        dot_product_one = tf.keras.layers.Dot(axes=-1)([candidate_one_vec, user_vec])
        pred_one_score = tf.keras.layers.Activation("sigmoid")(dot_product_one)

        # Create models
        model = tf.keras.Model(
            inputs=[history_input, candidate_input],
            outputs=pred_scores,
            name="nrms_model",
        )

        scorer = tf.keras.Model(
            inputs=[history_input, candidate_one_input],
            outputs=pred_one_score,
            name="nrms_scorer",
        )

        return model, scorer

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = True) -> tf.Tensor:
        """Forward pass using the appropriate model."""
        # Adapt input keys to match model expectations
        adapted_inputs = {
            "history_input": inputs["hist_tokens"],
        }

        if training:
            adapted_inputs["candidate_input"] = inputs["cand_tokens"]
            return self.model(adapted_inputs)

        # For validation/testing, reshape the candidates to process them all at once
        # Shape of cand_tokens: (batch=1, num_candidates, seq_length)
        _, num_candidates, seq_length = inputs["cand_tokens"].shape

        # Reshape candidates to (num_candidates, 1, seq_length)
        reshaped_candidates = tf.reshape(inputs["cand_tokens"], [num_candidates, 1, seq_length])

        # Repeat history for each candidate
        repeated_history = tf.repeat(inputs["hist_tokens"], repeats=num_candidates, axis=0)

        # Process all candidates at once
        adapted_inputs["history_input"] = repeated_history
        adapted_inputs["candidate_input_one_instance"] = reshaped_candidates
        scores = self.scorer(adapted_inputs)

        # Reshape scores to (1, num_candidates) to match expected output shape
        return tf.reshape(scores, [1, num_candidates])

from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras.layers import (
    Dropout,
    Layer,
    MultiHeadAttention,
    TimeDistributed,
)
from .layers import AdditiveSelfAttention

# from models.base import BaseNewsRecommender #TODO: add an abstract base class


class NewsEncoder(Layer):
    """News encoder with multi-head self-attention followed by additive attention."""

    def __init__(
        self,
        multiheads: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout_rate: float = 0.2,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super(NewsEncoder, self).__init__(**kwargs)
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout_rate = dropout_rate
        self.seed = seed

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer."""
        self.multihead_attention = MultiHeadAttention(
            num_heads=self.multiheads, key_dim=self.head_dim, seed=self.seed
        )
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)

        self.additive_attention = AdditiveSelfAttention(
            query_vector_dim=self.attention_hidden_dim, dropout=self.dropout_rate
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        # Input shape: (batch_size, seq_length, embedding_dim)
        # Output shape: (batch_size, head_dim * multiheads)
        return (input_shape[0], self.head_dim * self.multiheads)

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Process title embeddings."""
        # Apply dropout
        title_repr = self.dropout(inputs, training=training)

        # Multi-head self attention
        title_repr = self.multihead_attention(query=title_repr, key=title_repr, value=title_repr)

        # Apply dropout after attention
        title_repr = self.dropout(title_repr, training=training)

        # Additive attention
        return self.additive_attention(title_repr, training=training)


class UserEncoder(Layer):
    """User encoder with multi-head self-attention followed by additive attention."""

    def __init__(
        self,
        multiheads: int,
        head_dim: int,
        attention_hidden_dim: int,
        seed: int = 0,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super(UserEncoder, self).__init__(**kwargs)
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.seed = seed
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer."""
        # Multi-head self attention
        self.multihead_attention = MultiHeadAttention(
            num_heads=self.multiheads, key_dim=self.head_dim, seed=self.seed
        )

        # Additive attention for final user representation
        self.additive_attention = AdditiveSelfAttention(
            query_vector_dim=self.attention_hidden_dim, dropout=self.dropout_rate
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        # Input shape: (batch_size, history_length, news_vector_dim)
        # Output shape: (batch_size, head_dim * multiheads)
        return (input_shape[0], self.head_dim * self.multiheads)

    def call(self, news_vecs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Process news vectors."""
        # Multi-head self attention
        click_repr = self.multihead_attention(query=news_vecs, key=news_vecs, value=news_vecs)

        # Additive attention for final user representation
        return self.additive_attention(click_repr, training=training)


class NRMS(tf.keras.Model):
    def __init__(
        self,
        processed_news: Dict[str, Any],
        embedding_size: int = 300,
        multiheads: int = 16,
        head_dim: int = 64,
        attention_hidden_dim: int = 200,
        dropout_rate: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.vocab_size = processed_news["vocab_size"]
        self.embeddings = processed_news["embeddings"]

        # Build both models
        self.model, self.scorer = self._build_nrms()

    def _build_nrms(self):
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
        candidate_input_one_instance = tf.keras.Input(
            shape=(1, None), dtype="int32", name="candidate_input_one_instance"  # (1, title_length)
        )

        # Reshape single candidate input
        candidate_one_instance_reshape = tf.keras.layers.Reshape((-1,))(
            candidate_input_one_instance
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

        # Create user encoder that uses the news encoder
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
        candidate_one_embeds = embedding_layer(candidate_one_instance_reshape)
        candidate_one_vec = news_encoder(candidate_one_embeds)

        # Use dot product and sigmoid for scoring single candidate
        dot_product_one = tf.keras.layers.Dot(axes=-1)([candidate_one_vec, user_vec])
        pred_one_score = tf.keras.layers.Activation("sigmoid")(dot_product_one)

        # Create models
        model = tf.keras.Model(
            inputs=[history_input, candidate_input], outputs=pred_scores, name="nrms_model"
        )

        scorer = tf.keras.Model(
            inputs=[history_input, candidate_input_one_instance],
            outputs=pred_one_score,
            name="nrms_scorer",
        )

        return model, scorer

    def call(self, inputs, training=True):
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
        scores = tf.reshape(scores, [1, num_candidates])

        return scores

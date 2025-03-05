from typing import Any, Optional, Dict

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Layer,
    MultiHeadAttention,
    Softmax,
    TimeDistributed,
)

from models.base import BaseNewsRecommender

class NewsEncoder(Layer):
    """News encoder with multi-head self-attention."""

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

    def build(self, input_shape):
        """Build the layer."""
        # Multi-head self attention
        self.multihead_attention = MultiHeadAttention(
            num_heads=self.multiheads, 
            key_dim=self.head_dim, 
            seed=self.seed
        )

        # Additive attention
        self.attention_dense = Dense(
            self.attention_hidden_dim,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
        )
        self.attention_query = Dense(
            1, 
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
        )
        self.attention_softmax = Softmax(axis=1)

        # Dropout for regularization
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Process title embeddings."""
        # Apply dropout
        title_repr = self.dropout(inputs, training=training)

        # Multi-head self attention
        title_repr = self.multihead_attention(query=title_repr, key=title_repr, value=title_repr)

        # Apply dropout after attention
        title_repr = self.dropout(title_repr, training=training)

        # Additive attention
        attention_hidden = self.attention_dense(title_repr)
        attention_score = self.attention_query(attention_hidden)
        attention_weights = self.attention_softmax(attention_score)

        # Weighted sum to get final news vector
        news_vector = tf.reduce_sum(title_repr * attention_weights, axis=1)

        return news_vector


class UserEncoder(Layer):
    """User encoder with multi-head self-attention."""

    def __init__(
        self,
        multiheads: int,
        head_dim: int,
        attention_hidden_dim: int,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super(UserEncoder, self).__init__(**kwargs)

        # Multi-head self attention
        self.multihead_attention = MultiHeadAttention(
            num_heads=multiheads, key_dim=head_dim, seed=seed
        )

        # Additive attention
        self.attention_dense = Dense(
            attention_hidden_dim,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        )
        self.attention_query = Dense(
            1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
        self.attention_softmax = Softmax(axis=1)

    def call(self, news_vecs: tf.Tensor, attention_mask: tf.Tensor = None) -> tf.Tensor:
        """Process news vectors from history."""
        # Multi-head self attention with mask
        click_repr = self.multihead_attention(
            query=news_vecs, 
            key=news_vecs, 
            value=news_vecs,
            attention_mask=attention_mask[..., tf.newaxis]  # Add broadcast dim
        )

        # Additive attention
        attention_hidden = self.attention_dense(click_repr)
        attention_score = self.attention_query(attention_hidden)
        
        # Apply mask to attention scores
        if attention_mask is not None:
            attention_score = attention_score * tf.cast(
                attention_mask[..., tf.newaxis], attention_score.dtype
            )
            
        attention_weights = self.attention_softmax(attention_score)

        # Weighted sum to get final user vector
        user_vector = tf.reduce_sum(click_repr * attention_weights, axis=1)

        return user_vector


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
        
        # Use processed_news instead of dataset
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=processed_news["vocab_size"],
            output_dim=embedding_size,
            embeddings_initializer=tf.keras.initializers.Constant(
                processed_news["embeddings"]
            ),
            trainable=True,
            mask_zero=True
        )

        # News encoder wrapped in TimeDistributed
        self.news_encoder = TimeDistributed(
            NewsEncoder(
                multiheads=multiheads,
                head_dim=head_dim,
                attention_hidden_dim=attention_hidden_dim,
                dropout_rate=dropout_rate,
                seed=seed
            )
        )

        # User encoder
        self.user_encoder = UserEncoder(
            multiheads=multiheads,
            head_dim=head_dim,
            attention_hidden_dim=attention_hidden_dim,
            seed=seed
        )

    def call(self, inputs):
        # print("*******************")
        user_tokens = inputs["user_tokens"]  # Shape: (batch_size, num_history, title_length)
        # print("--- user_tokens ----")
        # print(user_tokens.shape)
        user_masks = inputs["user_masks"]    # Shape: (batch_size, num_history)
        # print("--- user_masks ----")
        # print(user_masks.shape)
        cand_tokens = inputs["cand_tokens"]  # Shape: (batch_size, num_candidates, title_length)
        # print("--- cand_tokens ----")
        # print(cand_tokens.shape)
        # Get embeddings using embedding layer
        user_embeds = self.embedding_layer(user_tokens)  # (batch, history, title_len, emb_dim)
        # print("--- user_embeds ----")
        # print(user_embeds.shape)
        cand_embeds = self.embedding_layer(cand_tokens)  # (batch, cand, title_len, emb_dim)
        # print("--- cand_embeds ----")
        # print(cand_embeds.shape)

        # Encode each news article independently
        user_news_vec = self.news_encoder(user_embeds)  # (batch, history, news_dim)
        # print("--- user_news_vec ----")
        # print(user_news_vec.shape)
        cand_news_vec = self.news_encoder(cand_embeds)  # (batch, cand, news_dim)
        # print("--- cand_news_vec ----")
        # print(cand_news_vec.shape)

        # Apply masks
        user_news_vec = user_news_vec * tf.cast(user_masks[..., tf.newaxis], user_news_vec.dtype)
        # print("--- user_news_vec after mask ----")
        # print(user_news_vec.shape)
        # Encode user from their history using attention mask
        user_vec = self.user_encoder(user_news_vec, attention_mask=user_masks)  # (batch, user_dim)
        # print("--- user_vec ----")
        # print(user_vec.shape)

        # Compute click probability
        scores = tf.einsum("bd,bcd->bc", user_vec, cand_news_vec)
        # print("--- scores ----")
        # print(scores.shape)

        return scores

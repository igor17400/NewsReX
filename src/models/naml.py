from typing import Any, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from .base import BaseNewsEncoder, BaseNewsRecommender, BaseUserEncoder
from .layers import AdditiveSelfAttention, MultiHeadSelfAttention


class NAMLNewsEncoder(BaseNewsEncoder):
    def __init__(
        self,
        word_embedding_dim: int,
        category_embedding_dim: int,
        cnn_filter_num: int,
        cnn_kernel_size: int,
        query_vector_dim: int,
        dense_units: int,
        dropout_rate: float,
        news_embedding_dim: int,
        max_vocabulary_size: int,
        num_categories: int,
        num_subcategories: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Word embedding
        self.word_embedding = layers.Embedding(
            input_dim=max_vocabulary_size,
            output_dim=word_embedding_dim,
        )

        # CNN layers for text
        self.title_cnn = layers.Conv1D(
            filters=cnn_filter_num,
            kernel_size=cnn_kernel_size,
            padding="same",
            activation="relu",
        )

        self.abstract_cnn = layers.Conv1D(
            filters=cnn_filter_num,
            kernel_size=cnn_kernel_size,
            padding="same",
            activation="relu",
        )

        # Category embeddings
        self.category_embedding = layers.Embedding(
            input_dim=num_categories,
            output_dim=category_embedding_dim,
        )

        self.subcategory_embedding = layers.Embedding(
            input_dim=num_subcategories,
            output_dim=category_embedding_dim,
        )

        # Attention
        self.attention = AdditiveSelfAttention(
            query_vector_dim=query_vector_dim, dropout=dropout_rate
        )

        # Final dense layers
        self.dense1 = layers.Dense(dense_units, activation="relu")
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(news_embedding_dim)

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        title, abstract, category, subcategory = inputs

        # Process title
        title_embedded = self.word_embedding(title)
        title_conv = self.title_cnn(title_embedded)
        title_vector = self.attention(title_conv, training=training)

        # Process abstract
        abstract_embedded = self.word_embedding(abstract)
        abstract_conv = self.abstract_cnn(abstract_embedded)
        abstract_vector = self.attention(abstract_conv, training=training)

        # Process categories
        category_vector = self.category_embedding(category)
        subcategory_vector = self.subcategory_embedding(subcategory)

        # Concatenate all features
        concat_vector = tf.concat(
            [title_vector, abstract_vector, category_vector, subcategory_vector], axis=1
        )

        # Final layers
        dense1 = self.dense1(concat_vector)
        dropout = self.dropout(dense1, training=training)
        news_vector = self.dense2(dropout)

        return news_vector


class NAMLUserEncoder(BaseUserEncoder):
    def __init__(
        self,
        num_attention_heads: int,
        query_vector_dim: int,
        dropout_rate: float,
        news_embedding_dim: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.multihead_attention = MultiHeadSelfAttention(
            num_heads=num_attention_heads,
            head_size=news_embedding_dim // num_attention_heads,
            dropout=dropout_rate,
        )

        self.additive_attention = AdditiveSelfAttention(
            query_vector_dim=query_vector_dim, dropout=dropout_rate
        )

    def call(self, news_vectors: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        attended_news = self.multihead_attention(news_vectors, training=training)
        user_vector = self.additive_attention(attended_news, training=training)
        return user_vector


class NAML(BaseNewsRecommender):
    def __init__(
        self,
        word_embedding_dim: int,
        category_embedding_dim: int,
        cnn_filter_num: int,
        cnn_kernel_size: int,
        num_attention_heads: int,
        query_vector_dim: int,
        dense_units: int,
        dropout_rate: float,
        news_embedding_dim: int,
        max_vocabulary_size: int,
        num_categories: int,
        num_subcategories: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.news_encoder = NAMLNewsEncoder(
            word_embedding_dim,
            category_embedding_dim,
            cnn_filter_num,
            cnn_kernel_size,
            query_vector_dim,
            dense_units,
            dropout_rate,
            news_embedding_dim,
            max_vocabulary_size,
            num_categories,
            num_subcategories,
        )
        self.user_encoder = NAMLUserEncoder(
            num_attention_heads, query_vector_dim, dropout_rate, news_embedding_dim
        )

    def encode_news(self, news_input: Any, training: Optional[bool] = None) -> tf.Tensor:
        return self.news_encoder(news_input, training=training)

    def encode_user(self, history_input: Any, training: Optional[bool] = None) -> tf.Tensor:
        # First encode all historical news
        history_vectors = self.news_encoder(history_input, training=training)
        # Then create user vector from historical news vectors
        return self.user_encoder(history_vectors, training=training)

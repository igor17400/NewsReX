from typing import Tuple, Dict, Any, Optional

import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow import keras
import math
import numpy as np

from .layers import AdditiveAttentionLayer, MAB, IntentDisentanglementLayer, CategoryPredictor, CustomGATLayer, BipartiteGraphCreator, UserAttentionLayer
from .base import BaseModel
from tensorflow.keras.regularizers import l2

class NewsInputSplitter(layers.Layer):
    """Custom layer to split concatenated news input into separate components.

    This layer handles the input format from the dataloader which provides
    concatenated features: [title_tokens, abstract_tokens, category, subcategory]
    """

    def __init__(self, title_length: int, abstract_length: int, **kwargs):
        super().__init__(**kwargs)
        self.title_length = title_length
        self.abstract_length = abstract_length

    def call(self, inputs):
        """Split concatenated input into separate components.

        Args:
            inputs: Concatenated tensor of shape (batch_size, title_length + abstract_length + 2)

        Returns:
            Dictionary with separated components
        """
        # Split the concatenated input
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


class CROWN(BaseModel):
    """CROWN: A Novel Approach to Comprehending Users' Preferences for Accurate Personalized News Recommendation.

    This implementation follows the original CROWN paper with:
    - GraphSAGE for user-news bipartite graph
    - Intent disentanglement with k-intent learning
    - Category-aware intent disentanglement
    - Title-body similarity computation
    - Category predictor for auxiliary loss
    """

    def __init__(
        self,
        processed_news: Dict[str, Any],
        embedding_size: int = 300,
        attention_dim: int = 200,
        intent_embedding_dim: int = 200,
        intent_num: int = 4,  # k for k-intent disentanglement
        category_embedding_dim: int = 100,
        subcategory_embedding_dim: int = 100,
        word_embedding_dim: int = 300,
        head_num: int = 8,
        feedforward_dim: int = 512,
        num_layers: int = 2,
        isab_num_heads: int = 4,
        isab_num_inds: int = 4,
        alpha: float = 0.1,  # Weight for auxiliary loss
        dropout_rate: float = 0.2,
        max_title_length: int = 30,
        max_abstract_length: int = 100,
        max_history_length: int = 50,
        max_impressions_length: int = 5,
        seed: int = 42,
        name: str = "crown",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Configurable parameters
        self.embedding_size = embedding_size
        self.attention_dim = attention_dim
        self.intent_embedding_dim = intent_embedding_dim
        self.intent_num = intent_num
        self.category_embedding_dim = category_embedding_dim
        self.subcategory_embedding_dim = subcategory_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.head_num = head_num
        self.feedforward_dim = feedforward_dim
        self.num_layers = num_layers
        self.isab_num_heads = isab_num_heads
        self.isab_num_inds = isab_num_inds
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length
        self.max_history_length = max_history_length
        self.max_impressions_length = max_impressions_length
        self.seed = seed

        tf.random.set_seed(self.seed)

        # Unpack processed data from the dataset
        self.vocab_size = processed_news["vocab_size"]
        self.num_categories = processed_news.get("num_categories", 0)
        self.num_subcategories = processed_news.get("num_subcategories", 0)
        self.embeddings_matrix = processed_news["embeddings"]

        # News embedding dimension
        self.news_embedding_dim = (
            intent_embedding_dim * 2 + category_embedding_dim + subcategory_embedding_dim
        )

        # Build components
        self.word_embedding_layer = self._build_word_embedding_layer()
        self.category_embedding_layer = self._build_category_embedding_layer()
        self.subcategory_embedding_layer = self._build_subcategory_embedding_layer()

        # CROWN-specific components
        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder()

        # Build final training and scoring models
        self.training_model, self.scorer_model = self._build_graph_models()

    def _build_word_embedding_layer(self) -> layers.Embedding:
        """Builds the word embedding layer for news content."""
        return layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.word_embedding_dim,
            embeddings_initializer=keras.initializers.Constant(self.embeddings_matrix),
            trainable=True,
            mask_zero=False,
            name="word_embedding",
        )

    def _build_category_embedding_layer(self) -> layers.Embedding:
        """Builds the category embedding layer."""
        return layers.Embedding(
            input_dim=self.num_categories + 1,
            output_dim=self.category_embedding_dim,
            trainable=True,
            name="category_embedding",
        )

    def _build_subcategory_embedding_layer(self) -> layers.Embedding:
        """Builds the subcategory embedding layer."""
        return layers.Embedding(
            input_dim=self.num_subcategories + 1,
            output_dim=self.subcategory_embedding_dim,
            trainable=True,
            name="subcategory_embedding",
        )

    def _build_newsencoder(self) -> keras.Model:
        """Builds the CROWN news encoder with MAB blocks as in the paper."""
        input_news = keras.Input(
            shape=(self.max_title_length + self.max_abstract_length + 2,),
            dtype="int32",
            name="news_input",
        )

        # Split the input into its components
        news_components = NewsInputSplitter(
            title_length=self.max_title_length,
            abstract_length=self.max_abstract_length,
            name="news_input_splitter",
        )(input_news)

        # --- Word embedding ---
        title_w = self.word_embedding_layer(news_components["title_tokens"])
        body_w = self.word_embedding_layer(news_components["abstract_tokens"])
        title_w = layers.Dropout(self.dropout_rate)(title_w)
        body_w = layers.Dropout(self.dropout_rate)(body_w)

        # --- MAB block for title and body ---
        mab_title = MAB(
            embed_dim=self.word_embedding_dim,
            num_heads=self.head_num,
            ff_dim=self.feedforward_dim,
            dropout_rate=self.dropout_rate,
            name="mab_title",
        )(title_w, title_w)

        mab_body = MAB(
            embed_dim=self.word_embedding_dim,
            num_heads=self.head_num,
            ff_dim=self.feedforward_dim,
            dropout_rate=self.dropout_rate,
            name="mab_body",
        )(body_w, body_w)

        # Mean pooling
        title_embedding = keras.ops.mean(mab_title, axis=1)
        body_embedding = keras.ops.mean(mab_body, axis=1)

        # --- Category Encoder ---
        category_encoder = self._build_category_embedding_layer()
        category_embedding = category_encoder(news_components["category_id"])
        category_embedding = keras.ops.squeeze(category_embedding, axis=1)

        # --- Subcategory Encoder ---
        subcategory_encoder = self._build_subcategory_embedding_layer()
        subcategory_embedding = subcategory_encoder(news_components["subcategory_id"])
        subcategory_embedding = keras.ops.squeeze(subcategory_embedding, axis=1)

        # --- Category-aware intent disentanglement ---
        concat_category = layers.Concatenate(axis=-1)([category_embedding, subcategory_embedding])
        category_representation = layers.Dense(
            self.category_embedding_dim, activation="relu", name="category_affine"
        )(concat_category)

        category_aware_title = layers.Concatenate(axis=-1)(
            [title_embedding, category_representation]
        )
        category_aware_body = layers.Concatenate(axis=-1)([body_embedding, category_representation])

        # Intent disentanglement
        intent_disentangle = IntentDisentanglementLayer(
            self.intent_num, self.intent_embedding_dim, self.dropout_rate
        )

        title_k_intent = intent_disentangle(category_aware_title)
        body_k_intent = intent_disentangle(category_aware_body)

        # Intent-based attention
        title_intent_attention = AdditiveAttentionLayer(
            self.attention_dim, name="title_intent_attention"
        )
        body_intent_attention = AdditiveAttentionLayer(
            self.attention_dim, name="body_intent_attention"
        )

        title_intent_embedding = title_intent_attention(title_k_intent)
        body_intent_embedding = body_intent_attention(body_k_intent)

        # Category predictor for auxiliary loss
        category_predictor = CategoryPredictor(self.intent_embedding_dim, self.num_categories)
        target_category = keras.ops.cast(news_components["category_id"], "int32")
        category_loss = category_predictor(title_intent_embedding, target_category)

        # Store auxiliary loss
        self.auxiliary_loss = category_loss * self.alpha

        # Title-body similarity computation
        title_body_similarity = keras.ops.sum(
            title_intent_embedding * body_intent_embedding, axis=-1, keepdims=True
        )
        title_body_similarity = (title_body_similarity + 1.0) / 2.0

        # Final news representation
        news_representation = keras.ops.concatenate(
            [title_intent_embedding, title_body_similarity * body_intent_embedding], axis=-1
        )

        # Feature fusion with category and subcategory
        news_representation = keras.ops.concatenate(
            [news_representation, category_embedding, subcategory_embedding], axis=-1
        )

        return keras.Model(input_news, news_representation, name="news_encoder")

    def _build_userencoder(self) -> keras.Model:
        """Builds the CROWN user encoder with custom GAT layers.
        Expects history_input and creates bipartite graph internally using history mask.
        """
        history_input = keras.Input(
            shape=(self.max_history_length, self.max_title_length + self.max_abstract_length + 2),
            dtype="int32",
            name="history_input",
        )

        # Process each news item in history through news encoder
        news_vectors = layers.TimeDistributed(self.newsencoder, name="td_history_news_encoder")(
            history_input
        )

        # Create history mask by checking for non-zero values in the first token position
        # This assumes padding tokens are 0
        history_mask = layers.Lambda(
            lambda x: keras.ops.not_equal(keras.ops.sum(x, axis=-1), 0),
            name="create_history_mask"
        )(history_input)

        # Create bipartite graph adjacency matrix
        bipartite_creator = BipartiteGraphCreator(self.max_history_length)
        adjacency_matrix = bipartite_creator(history_mask)

        # First GAT layer
        gat_layer_1 = CustomGATLayer(
            output_dim=self.news_embedding_dim,
            num_heads=8,
            dropout_rate=self.dropout_rate,
            activation="elu",
            kernel_regularizer=l2(2.5e-4),
            bias_regularizer=l2(2.5e-4),
            concat_heads=True,
            name="gat_layer_1"
        )
        gat_features_1 = gat_layer_1([news_vectors, adjacency_matrix])

        # Second GAT layer
        gat_layer_2 = CustomGATLayer(
            output_dim=self.news_embedding_dim,
            num_heads=1,
            dropout_rate=self.dropout_rate,
            activation="relu",
            kernel_regularizer=l2(2.5e-4),
            bias_regularizer=l2(2.5e-4),
            concat_heads=False,
            name="gat_layer_2"
        )
        gat_features = gat_layer_2([gat_features_1, adjacency_matrix])

        # Attention mechanism for user representation (matching original PyTorch implementation)
        user_attention = UserAttentionLayer(
            attention_dim=self.attention_dim,
            news_embedding_dim=self.news_embedding_dim,
            name="user_attention_layer"
        )
        user_representation = user_attention([gat_features, history_mask])

        return keras.Model(
            history_input,
            user_representation, 
            name="user_encoder"
        )

    def _build_graph_models(self) -> Tuple[keras.Model, keras.Model]:
        """Builds the main training and scoring models
        Accepts separate input tensors for title, abstract, category, subcategory for both history and candidates, plus edge_index_input.
        Concatenates inside the model before passing to encoders.
        """
        # --- Inputs for Training Model ---
        hist_title_tokens = keras.Input(
            shape=(self.max_history_length, self.max_title_length),
            dtype="int32",
            name="hist_title_tokens",
        )
        hist_abstract_tokens = keras.Input(
            shape=(self.max_history_length, self.max_abstract_length),
            dtype="int32",
            name="hist_abstract_tokens",
        )
        hist_category = keras.Input(
            shape=(self.max_history_length, 1), dtype="int32", name="hist_category"
        )
        hist_subcategory = keras.Input(
            shape=(self.max_history_length, 1), dtype="int32", name="hist_subcategory"
        )
        cand_title_tokens = keras.Input(
            shape=(self.max_impressions_length, self.max_title_length),
            dtype="int32",
            name="cand_title_tokens",
        )
        cand_abstract_tokens = keras.Input(
            shape=(self.max_impressions_length, self.max_abstract_length),
            dtype="int32",
            name="cand_abstract_tokens",
        )
        cand_category = keras.Input(
            shape=(self.max_impressions_length, 1), dtype="int32", name="cand_category"
        )
        cand_subcategory = keras.Input(
            shape=(self.max_impressions_length, 1), dtype="int32", name="cand_subcategory"
        )

        # Concatenate history and candidate features
        history_concat = layers.Concatenate(axis=-1)(
            [hist_title_tokens, hist_abstract_tokens, hist_category, hist_subcategory]
        )
        candidate_concat = layers.Concatenate(axis=-1)(
            [cand_title_tokens, cand_abstract_tokens, cand_category, cand_subcategory]
        )

        # --- Training Model Graph ---
        user_representation_train = self.userencoder(history_concat)
        candidate_news_representation_train = layers.TimeDistributed(
            self.newsencoder, name="td_candidate_news_encoder_train"
        )(candidate_concat)

        scores_train = layers.Dot(axes=-1, name="dot_product_train")(
            [candidate_news_representation_train, user_representation_train]
        )
        preds_train = layers.Activation("softmax", name="softmax_activation_train")(scores_train)

        # --- Training Model ---
        training_model = keras.Model(
            inputs=[history_concat, candidate_concat],
            outputs=preds_train,
            name="crown_training_model",
        )

        # --- Scoring Model Graph ---
        user_hist_title_tokens = keras.Input(
            shape=(self.max_history_length, self.max_title_length),
            dtype="int32",
            name="user_hist_title_tokens",
        )
        user_hist_abstract_tokens = keras.Input(
            shape=(self.max_history_length, self.max_abstract_length),
            dtype="int32",
            name="user_hist_abstract_tokens",
        )
        user_hist_category = keras.Input(
            shape=(self.max_history_length, 1), dtype="int32", name="user_hist_category"
        )
        user_hist_subcategory = keras.Input(
            shape=(self.max_history_length, 1), dtype="int32", name="user_hist_subcategory"
        )
        news_title_tokens = keras.Input(
            shape=(self.max_title_length,), dtype="int32", name="news_title_tokens"
        )
        news_abstract_tokens = keras.Input(
            shape=(self.max_abstract_length,), dtype="int32", name="news_abstract_tokens"
        )
        news_category = keras.Input(shape=(1,), dtype="int32", name="news_category")
        news_subcategory = keras.Input(shape=(1,), dtype="int32", name="news_subcategory")
        user_history_concat = layers.Concatenate(axis=-1)(
            [
                user_hist_title_tokens,
                user_hist_abstract_tokens,
                user_hist_category,
                user_hist_subcategory,
            ]
        )
        news_concat = layers.Concatenate(axis=-1)(
            [news_title_tokens, news_abstract_tokens, news_category, news_subcategory]
        )
        user_repr = self.userencoder(user_history_concat)
        news_repr = self.newsencoder(news_concat)
        score = keras.ops.sum(user_repr * news_repr, axis=-1)
        scorer_model = keras.Model(
            inputs=[user_history_concat, news_concat],
            outputs=score,
            name="crown_scorer_model",
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
                - 'edge_index': Graph edge indices for GraphSAGE

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
        edge_index = inputs.get("edge_index", None)

        # Reshape category and subcategory tensors to match the expected input shape
        hist_category = keras.ops.expand_dims(hist_category, axis=-1)
        hist_subcategory = keras.ops.expand_dims(hist_subcategory, axis=-1)
        cand_category = keras.ops.expand_dims(cand_category, axis=-1)
        cand_subcategory = keras.ops.expand_dims(cand_subcategory, axis=-1)

        # Concatenate features for history
        hist_features = keras.ops.concatenate(
            [hist_title_tokens, hist_abstract_tokens, hist_category, hist_subcategory], axis=-1
        )

        # Concatenate features for candidates
        cand_features = keras.ops.concatenate(
            [cand_title_tokens, cand_abstract_tokens, cand_category, cand_subcategory], axis=-1
        )

        # Handle edge_index for GraphSAGE
        if edge_index is not None:
            return self.training_model(
                [hist_features, cand_features, edge_index],
                training=True,
            )
        else:
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
            "Invalid input format for CROWN inference. Expected 'candidate_news_tokens'"
        )

    def _score_multiple_candidates(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Scores multiple candidates for each history in the batch by iterating."""
        history_batch = {k: v for k, v in inputs.items() if k.startswith("history_")}
        candidates_batch = {k: v for k, v in inputs.items() if k.startswith("candidate_")}
        edge_index = inputs.get("edge_index", None)

        batch_size = tf.shape(next(iter(history_batch.values())))[0]
        num_candidates = tf.shape(next(iter(candidates_batch.values())))[1]

        all_scores = []
        for i in keras.ops.arange(batch_size):
            current_history = {k: keras.ops.expand_dims(v[i], 0) for k, v in history_batch.items()}
            if edge_index is not None:
                current_edge_index = keras.ops.expand_dims(edge_index[i], 0)
                current_history["edge_index"] = current_edge_index

            candidate_scores = []
            for j in keras.ops.arange(num_candidates):
                current_candidate = {k: v[i, j] for k, v in candidates_batch.items()}
                score = self.scorer_model({**current_history, **current_candidate})
                candidate_scores.append(score)

            item_scores = keras.ops.concatenate(candidate_scores, axis=1)
            all_scores.append(item_scores)

        return keras.ops.concatenate(all_scores, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_size": self.embedding_size,
                "attention_dim": self.attention_dim,
                "intent_embedding_dim": self.intent_embedding_dim,
                "intent_num": self.intent_num,
                "category_embedding_dim": self.category_embedding_dim,
                "subcategory_embedding_dim": self.subcategory_embedding_dim,
                "word_embedding_dim": self.word_embedding_dim,
                "head_num": self.head_num,
                "feedforward_dim": self.feedforward_dim,
                "num_layers": self.num_layers,
                "isab_num_heads": self.isab_num_heads,
                "isab_num_inds": self.isab_num_inds,
                "alpha": self.alpha,
                "dropout_rate": self.dropout_rate,
                "max_title_length": self.max_title_length,
                "max_abstract_length": self.max_abstract_length,
                "max_history_length": self.max_history_length,
                "max_impressions_length": self.max_impressions_length,
                "seed": self.seed,
                "vocab_size": self.vocab_size,
                "num_categories": self.num_categories,
                "num_subcategories": self.num_subcategories,
            }
        )
        return config

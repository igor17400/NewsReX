from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

# Assuming AdditiveSelfAttention is correctly defined in .layers
from .layers import AdditiveSelfAttention
import numpy as np

# from utils.losses import get_loss # Not used in this file directly
from utils.io import save_predictions_to_file_fn


class SelfAttention(layers.Layer):
    """Multi-head self attention layer using Keras's built-in implementation.

    This layer is a wrapper around Keras's MultiHeadAttention that simplifies self-attention usage.
    Key differences from raw MultiHeadAttention:
    1. Simplified Interface:
        - Automatically uses same tensor for query, key, and value (self-attention)
        - No need to manually specify q,k,v when they're the same tensor

    2. Fixed Output Shape:
        - Ensures output matches input dimensions
        - Handles dimension management internally

    3. Built-in Features:
        - Includes dropout for regularization
        - Handles layer normalization
        - Manages attention head dimensions

    Usage in NRMS:
    1. Word-level attention: Captures relationships between words in news titles
    2. News-level attention: Models relationships between news articles in user history

    Example:
        # For word embeddings in a title
        word_embeddings = ... # shape: (batch_size, seq_len, embed_dim)
        attention_output = SelfAttention(multiheads=8, head_dim=64)(word_embeddings)
        # Output shape: (batch_size, seq_len, multiheads * head_dim)
    """

    def __init__(self, multiheads, head_dim, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.seed = seed
        self.multihead_attention = None

    def build(self, input_shape):
        """Initializes the internal MultiHeadAttention layer.

        This method creates the MultiHeadAttention layer with the specified number of heads, head dimension, and seed.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Create the MultiHeadAttention layer
        self.multihead_attention = layers.MultiHeadAttention(
            num_heads=self.multiheads,
            key_dim=self.head_dim,
            value_dim=self.head_dim,  # value_dim should typically match key_dim
            output_shape=self.head_dim
            * self.multiheads,  # output_shape is inferred, not set directly. This might be an old API usage or custom expectation.
            # For standard MHA, output is (batch_size, seq_len, num_heads * key_dim) if output_shape is None.
            # If output_shape is specified, it dictates the last dimension.
            seed=self.seed,
        )
        super().build(input_shape)

    def call(
        self, inputs, training=None
    ):  # Added training argument for consistency, MHA handles it
        """Applies multi-head self-attention to the input tensor(s).

        This method computes self-attention over the input(s) using the underlying Keras MultiHeadAttention layer.

        Args:
            inputs: Either a single tensor (used as query, key, and value) or a list of three tensors [query, key, value].
            training: Boolean or None. Indicates whether the layer should behave in training mode or inference mode.

        Returns:
            tf.Tensor: The result of the multi-head attention operation.
        """
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        # Pass the training argument to the underlying MHA layer
        return self.multihead_attention(
            query, value, key, training=training
        )  # Correct order for MHA is query, value, key

    def get_config(self):
        """Returns the configuration of the SelfAttention layer for serialization.

        This method provides a dictionary of the layer's hyperparameters and settings,
        enabling the layer to be saved and reconstructed later.

        Returns:
            dict: Layer configuration including number of heads, head dimension, and seed.
        """
        config = super().get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "seed": self.seed,
            }
        )
        return config


class NRMS(tf.keras.Model):
    """Neural News Recommendation with Multi-Head Self-Attention (NRMS) model."""

    def __init__(
        self,
        processed_news,
        embedding_size=300,
        multiheads=16,
        head_dim=16,
        attention_hidden_dim=200,
        dropout_rate=0.2,
        seed=42,
        name="nrms",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.embedding_size = embedding_size
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout_rate = dropout_rate
        self.seed = seed

        tf.random.set_seed(
            self.seed
        )  # Set global TF seed for reproducibility if needed elsewhere too

        self.vocab_size = processed_news["vocab_size"]
        self.embeddings_matrix = processed_news["embeddings"]

        # Build components
        self.embedding_layer = layers.Embedding(
            self.vocab_size,
            self.embedding_size,
            embeddings_initializer=tf.keras.initializers.Constant(self.embeddings_matrix),
            trainable=True,
            mask_zero=False,
            name="word_embedding",
        )

        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder()

        # Build the main training model and the scorer model
        self.training_model, self.scorer_model = self._build_graph_models()

    def _build_newsencoder(self) -> tf.keras.Model:
        """Build the news encoder component.

        The news encoder processes news titles through:
        1. Word embeddings
        2. Dropout for regularization
        3. Word-level self-attention to capture important words
        4. Additive attention to get a single news vector

        Returns:
            tf.keras.Model: The news encoder model
                Input: (batch_size, title_length) - tokenized news titles
                Output: (batch_size, embedding_dim) - news embeddings
        """
        sequences_input_title = tf.keras.Input(
            shape=(None,), dtype="int32", name="news_tokens_input"
        )

        # Word Embedding
        embedded_sequences_title = self.embedding_layer(sequences_input_title)

        # Dropout after embedding
        y = layers.Dropout(self.dropout_rate, seed=self.seed)(embedded_sequences_title)

        # Self-Attention over words in a title
        y = SelfAttention(
            self.multiheads, self.head_dim, seed=self.seed, name="title_word_self_attention"
        )([y, y, y])
        y = layers.Dropout(self.dropout_rate, seed=self.seed + 1)(y)

        # Additive Attention to get a single news vector
        pred_title = AdditiveSelfAttention(
            self.attention_hidden_dim, name="title_additive_attention"
        )(y)

        return tf.keras.Model(sequences_input_title, pred_title, name="news_encoder")

    def _build_userencoder(self) -> tf.keras.Model:
        """Build the user encoder component.

        The user encoder processes user history through:
        1. News encoder to get embeddings for each news in history
        2. News-level self-attention to model user interests
        3. Additive attention to get a single user vector

        Returns:
            tf.keras.Model: The user encoder model
                Input: (batch_size, history_length, title_length) - tokenized news history
                Output: (batch_size, embedding_dim) - user embeddings
        """
        # Input shape: (batch_size, num_history_news, num_words_in_title)
        his_input_title_tokens = tf.keras.Input(
            shape=(None, None), dtype="int32", name="history_news_tokens_input"
        )

        # Apply newsencoder to each news item in history
        click_title_presents = layers.TimeDistributed(
            self.newsencoder, name="time_distributed_news_encoder"
        )(his_input_title_tokens)

        # Self-Attention over browsed news
        y = SelfAttention(
            self.multiheads, self.head_dim, seed=self.seed + 2, name="browsed_news_self_attention"
        )([click_title_presents, click_title_presents, click_title_presents])
        y = layers.Dropout(self.dropout_rate, seed=self.seed + 3)(y)

        # Additive Attention to get a single user vector
        user_present = AdditiveSelfAttention(
            self.attention_hidden_dim, name="user_additive_attention"
        )(y)

        return tf.keras.Model(his_input_title_tokens, user_present, name="user_encoder")

    def _build_graph_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
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
        history_tokens_input_train = tf.keras.Input(
            shape=(None, None), dtype="int32", name="history_tokens_train"
        )
        candidate_tokens_input_train = tf.keras.Input(
            shape=(None, None), dtype="int32", name="candidate_tokens_train"
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

        training_model = tf.keras.Model(
            inputs=[history_tokens_input_train, candidate_tokens_input_train],
            outputs=preds_train,
            name="nrms_training_model",
        )

        # ------ Inputs for Scorer Model ------
        history_tokens_input_score = tf.keras.Input(
            shape=(None, None), dtype="int32", name="history_tokens_score"
        )
        single_candidate_tokens_input_score = tf.keras.Input(
            shape=(None,), dtype="int32", name="single_candidate_tokens_score"
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

        scorer_model = tf.keras.Model(
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
            tf.Tensor: Model predictions
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
            tf.Tensor: Softmax probabilities for each candidate
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
            tf.Tensor: Model predictions
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
        """Score multiple candidates for each history in the batch.

        This method processes a batch of user histories and their corresponding candidate news articles.
        For each user history, it scores all candidates against that history using the scorer_model.
        The scoring is done one candidate at a time to maintain batch consistency.

        Args:
            inputs (dict): Dictionary containing:
                - 'hist_tokens': User history tokens, shape (batch_size, history_len, title_len)
                - 'cand_tokens': Candidate news tokens, shape (batch_size, num_candidates, title_len)

        Returns:
            tf.Tensor: Scores for all candidates
                Shape: (batch_size, num_candidates)
                Each row contains the scores for all candidates for one user history

        Note:
            This method processes candidates one at a time to ensure proper batch handling
            and to maintain consistency with the scorer_model's expectations.
        """
        history_batch = inputs["hist_tokens"]  # Shape: (batch_size, history_len, title_len)
        candidates_batch = inputs["cand_tokens"]  # Shape: (batch_size, num_candidates, title_len)

        batch_size = tf.shape(history_batch)[0]
        num_candidates = tf.shape(candidates_batch)[1]

        all_scores = []

        # Process each item in the batch
        for i in range(batch_size):
            # Get history for current item
            current_history = tf.expand_dims(
                history_batch[i], 0
            )  # Shape: (1, history_len, title_len)

            # Score each candidate against this history
            candidate_scores = []
            for j in range(num_candidates):
                current_candidate = tf.expand_dims(
                    candidates_batch[i, j], 0
                )  # Shape: (1, title_len)
                score = self.scorer_model([current_history, current_candidate], training=False)
                candidate_scores.append(score)

            # Combine scores for all candidates of this item
            item_scores = tf.concat(candidate_scores, axis=1)  # Shape: (1, num_candidates)
            all_scores.append(item_scores)

        # Combine scores for all items in batch
        return tf.concat(all_scores, axis=0)  # Shape: (batch_size, num_candidates)

    def _precompute_news_vectors(
        self,
        processed_news_tokens: np.ndarray,
        news_ids_for_precompute: np.ndarray,
        batch_size_eval: int,
        progress: Any,
        progress_task: Any,
    ) -> Dict[str, np.ndarray]:
        """Precompute news vectors for efficient evaluation.

        This method processes news tokens in batches to generate embeddings for each news article.
        The embeddings are stored in a dictionary for efficient lookup during evaluation.

        Args:
            processed_news_tokens: Array of tokenized news titles
                Shape: (num_news, title_length)
            news_ids_for_precompute: Array of news IDs corresponding to the tokens
                Shape: (num_news,)
            batch_size_eval: Number of news items to process in each batch

        Returns:
            Dict mapping news IDs to their embeddings
                Keys: News IDs in format "N{id}" (e.g., "N123")
                Values: News embeddings of shape (embedding_dim,)
        """
        news_encoder_instance = self.newsencoder
        all_news_tokens = np.array(processed_news_tokens)
        num_news = len(all_news_tokens)
        news_vecs_dict = {}

        for i in range(0, num_news, batch_size_eval):
            # Process news in batches
            batch_tokens = all_news_tokens[i : i + batch_size_eval]
            batch_ids = news_ids_for_precompute[i : i + batch_size_eval]

            # Generate embeddings for the current batch
            batch_vecs = news_encoder_instance(
                tf.constant(batch_tokens, dtype=tf.int32), training=False
            )

            # Store embeddings in dictionary with news IDs as keys
            for news_id, vec in zip(batch_ids, batch_vecs.numpy()):
                # Handle news IDs that might already have 'N' prefix
                if isinstance(news_id, str) and news_id.startswith("N"):
                    news_id_key = news_id
                else:
                    news_id_key = f"N{int(news_id)}"
                news_vecs_dict[news_id_key] = vec

            progress.update(progress_task, advance=len(batch_tokens))

        return news_vecs_dict

    def _precompute_user_vectors(
        self,
        histories_news_tokens: np.ndarray,
        batch_size_eval: int,
        progress: Any,
        progress_task: Any,
    ) -> np.ndarray:
        """Precompute user vectors for efficient evaluation.

        This method processes user history tokens in batches to generate user embeddings.
        Each user embedding represents their interest profile based on their news reading history.

        Args:
            histories_news_tokens: Array of tokenized news titles in user histories
                Shape: (num_users, history_length, title_length)
            batch_size_eval: Number of users to process in each batch

        Returns:
            Array of user embeddings
                Shape: (num_users, embedding_dim)
        """
        user_encoder_instance = self.userencoder
        user_vectors_list = []
        num_users = len(histories_news_tokens)

        for i in range(0, num_users, batch_size_eval):
            # Process users in batches
            current_batch_history_tokens = histories_news_tokens[i : i + batch_size_eval]

            # Convert to tensor and generate user embeddings
            current_batch_history_tokens_tf = tf.constant(
                current_batch_history_tokens, dtype=tf.int32
            )
            user_vecs_batch = user_encoder_instance(current_batch_history_tokens_tf, training=False)

            # Store batch of user vectors
            user_vectors_list.append(user_vecs_batch.numpy())
            progress.update(progress_task, advance=len(current_batch_history_tokens))

        return np.concatenate(user_vectors_list, axis=0)

    def _compute_impression_scores(
        self,
        cand_ids: List[int],
        current_user_vector: np.ndarray,
        news_vectors_dict: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scores for a single impression's candidates.

        Args:
            cand_ids: List of candidate news IDs
            current_user_vector: User embedding vector
            news_vectors_dict: Dictionary of precomputed news vectors

        Returns:
            Tuple of (scores, labels) for the impression
        """
        candidate_vecs = []
        for nid in cand_ids:
            nid_str = f"N{int(nid)}"
            if nid_str in news_vectors_dict:
                candidate_vecs.append(news_vectors_dict[nid_str])
            else:
                embedding_dim = next(iter(news_vectors_dict.values())).shape[0]
                candidate_vecs.append(np.zeros(embedding_dim, dtype=np.float32))

        current_candidate_news_vectors = np.stack(candidate_vecs, axis=0)

        # Compute scores
        return np.dot(current_candidate_news_vectors, current_user_vector)

    def _compute_metrics(
        self,
        group_labels_list: List[np.ndarray],
        group_preds_list: List[np.ndarray],
        metrics_calculator: Any,
        progress: Any,
    ) -> Dict[str, float]:
        """Compute final metrics from collected predictions and labels.

        Args:
            group_labels_list: List of label arrays
            group_preds_list: List of prediction arrays
            metrics_calculator: Metrics calculator instance
            progress: Progress bar manager

        Returns:
            Dictionary of computed metrics
        """
        val_loss_total = 0
        num_valid_impressions_for_loss = 0
        metric_values_agg = {key: [] for key in metrics_calculator.METRIC_NAMES}

        for labels_np, scores_np in zip(group_labels_list, group_preds_list):
            if (
                labels_np.size == 0
                or scores_np.size == 0
                or labels_np.ndim == 0
                or scores_np.ndim == 0
            ):
                continue

            labels_tf = tf.constant([labels_np], dtype=tf.float32)
            scores_tf_logits = tf.constant([scores_np], dtype=tf.float32)
            scores_tf_probs_for_loss = tf.nn.softmax(scores_tf_logits, axis=-1)

            if self._has_nan_or_inf(scores_tf_probs_for_loss, labels_tf):
                progress.console.print(
                    f"[WARNING fast_evaluate] NaN/Inf detected in scores/labels. Skipping loss. Scores: {scores_np}, Labels: {labels_np}"
                )
                continue

            try:
                current_loss = self.compute_loss(
                    x=None,
                    y=labels_tf,
                    y_pred=scores_tf_probs_for_loss,
                    sample_weight=None,
                    training=False,
                ).numpy()
                val_loss_total += current_loss
                num_valid_impressions_for_loss += 1
            except Exception as e:
                progress.console.print(
                    f"[WARNING fast_evaluate] Error calculating loss: {e}. Scores: {scores_np}, Labels: {labels_np}"
                )

            impression_metrics = metrics_calculator.compute_metrics(
                y_true=labels_tf,
                y_pred_logits=scores_tf_logits,
                progress=None,
            )

            for metric_name, value in impression_metrics.items():
                if metric_name in metric_values_agg:
                    metric_values_agg[metric_name].append(float(value))

        final_metrics = {
            "loss": (
                (val_loss_total / num_valid_impressions_for_loss)
                if num_valid_impressions_for_loss > 0
                else 0.0
            )
        }
        for metric_name, values_list in metric_values_agg.items():
            final_metrics[metric_name] = sum(values_list) / len(values_list) if values_list else 0.0

        return final_metrics

    def _has_nan_or_inf(self, scores: tf.Tensor, labels: tf.Tensor) -> bool:
        """Check if tensors contain NaN or Inf values."""
        return (
            tf.reduce_any(tf.math.is_nan(scores))
            or tf.reduce_any(tf.math.is_inf(scores))
            or tf.reduce_any(tf.math.is_nan(labels))
            or tf.reduce_any(tf.math.is_inf(labels))
        )

    def fast_evaluate(
        self,
        behaviors_data: Dict[str, Any],
        processed_news: Dict[str, Any],
        metrics_calculator: Any,
        progress: Any,
        mode: str = "validate",
        save_predictions_path: Optional[Path] = None,
        epoch: Optional[int] = None,
        batch_size_eval: int = 128,
    ) -> Dict[str, float]:
        """Fast evaluation of the model using precomputed vectors."""
        num_impressions = len(behaviors_data["labels"])
        val_progress = progress.add_task(f"Running {mode}...", total=num_impressions, visible=True)

        # Precompute vectors with progress bars
        news_progress = progress.add_task(
            "Computing news vectors...", total=len(processed_news["tokens"]), visible=True
        )
        news_vectors_dict = self._precompute_news_vectors(
            processed_news["tokens"],
            processed_news["news_ids_original_strings"],
            batch_size_eval,
            progress,
            news_progress,
        )
        progress.remove_task(news_progress)

        user_progress = progress.add_task(
            "Computing user vectors...",
            total=len(behaviors_data["history_news_tokens"]),
            visible=True,
        )
        all_user_history_vectors = self._precompute_user_vectors(
            behaviors_data["history_news_tokens"],
            batch_size_eval,
            progress,
            user_progress,
        )
        progress.remove_task(user_progress)

        # Process impressions
        group_labels_list = []
        group_preds_list = []
        predictions_to_save = {}

        for idx in range(num_impressions):
            cand_ids = behaviors_data["candidate_news_ids"][idx]
            current_user_vector = all_user_history_vectors[idx]

            scores = self._compute_impression_scores(
                cand_ids, current_user_vector, news_vectors_dict
            )
            labels = np.array(behaviors_data["labels"][idx])

            group_labels_list.append(labels)
            group_preds_list.append(scores)

            if save_predictions_path is not None:
                predictions_to_save[str(idx)] = (labels.tolist(), scores.tolist())

            progress.update(val_progress, advance=1)

        # Compute final metrics
        final_metrics = self._compute_metrics(
            group_labels_list, group_preds_list, metrics_calculator, progress
        )
        final_metrics["num_impressions"] = num_impressions

        if save_predictions_path is not None:
            save_predictions_to_file_fn(predictions_to_save, save_predictions_path, epoch, mode)

        return final_metrics

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
                # "embeddings_matrix": self.embeddings_matrix.tolist() # Careful with large matrices in config
            }
        )
        # Note: processed_news is not part of the config as it's data.
        # Keras will save/load weights of sub-models (embedding_layer, newsencoder, userencoder, training_model, scorer_model)
        return config

import keras
import jax
import jax.numpy as jnp

import logging
from functools import partial

# Setup logger
logger = logging.getLogger(__name__)


class NewsRecommenderMetrics:
    """JAX-optimized metrics for evaluating news recommendation models."""

    METRIC_NAMES = ["auc", "mrr", "ndcg@5", "ndcg@10"]

    def __init__(self):
        """Initialize the JAX-optimized metrics calculator."""
        # JIT compile all metric functions for maximum performance
        self._compute_auc_jax = jax.jit(self._compute_auc_impl)
        self._compute_mrr_jax = jax.jit(self._compute_mrr_impl)
        self._compute_ndcg_jax = jax.jit(self._compute_ndcg_impl, static_argnames=['k'])
        self._dcg_score_jax = jax.jit(self._dcg_score_impl, static_argnames=['k'])

    def compute_metrics(self, y_true, y_pred_logits, progress=None):
        """
        Compute metrics for a single impression or batch using JAX.

        Args:
            y_true: Keras tensor, shape (batch_size, num_candidates) or (num_candidates,)
            y_pred_logits: Keras tensor, shape (batch_size, num_candidates) or (num_candidates,)
            progress: Optional progress bar for logging

        Returns:
            dict: Dictionary of metric names and their values
        """
        # Convert to JAX arrays
        y_true = jnp.asarray(keras.ops.convert_to_numpy(y_true))
        y_pred_logits = jnp.asarray(keras.ops.convert_to_numpy(y_pred_logits))

        if y_true.ndim == 1:
            # Single impression
            return self._compute_metrics_single(y_true, y_pred_logits)

        # Batch processing using vectorized JAX operations
        auc_scores = jax.vmap(self._compute_auc_jax)(y_true, y_pred_logits)
        mrr_scores = jax.vmap(self._compute_mrr_jax)(y_true, y_pred_logits)
        ndcg5_scores = jax.vmap(partial(self._compute_ndcg_jax, k=5))(y_true, y_pred_logits)
        ndcg10_scores = jax.vmap(partial(self._compute_ndcg_jax, k=10))(y_true, y_pred_logits)

        return {
            "auc": float(jnp.mean(auc_scores)),
            "mrr": float(jnp.mean(mrr_scores)),
            "ndcg@5": float(jnp.mean(ndcg5_scores)),
            "ndcg@10": float(jnp.mean(ndcg10_scores)),
        }

    def compute_metrics_from_scores(self, y_true_grouped, y_pred_scores_grouped, progress=None):
        """
        Compute metrics from grouped scores (for compatibility with existing code).

        Args:
            y_true_grouped: Keras tensor, shape (batch_size, num_candidates)
            y_pred_scores_grouped: Keras tensor, shape (batch_size, num_candidates)
            progress: Optional progress bar for logging

        Returns:
            dict: Dictionary of metric names and their values
        """
        return self.compute_metrics(y_true_grouped, y_pred_scores_grouped, progress)

    def _compute_metrics_single(self, y_true, y_pred_logits):
        """JAX-optimized metrics computation for a single impression."""
        auc = self._compute_auc_jax(y_true, y_pred_logits)
        mrr = self._compute_mrr_jax(y_true, y_pred_logits)
        ndcg5 = self._compute_ndcg_jax(y_true, y_pred_logits, k=5)
        ndcg10 = self._compute_ndcg_jax(y_true, y_pred_logits, k=10)

        return {
            "auc": float(auc),
            "mrr": float(mrr),
            "ndcg@5": float(ndcg5),
            "ndcg@10": float(ndcg10),
        }

    def _compute_auc_impl(self, y_true, y_pred):
        """JAX implementation of AUC calculation."""
        # Simple AUC implementation using ranking
        n_pos = jnp.sum(y_true)
        n_neg = jnp.sum(1 - y_true)

        # Handle edge cases
        def compute_auc():
            # Get indices sorted by prediction scores (descending)
            sorted_indices = jnp.argsort(-y_pred)
            y_true_sorted = y_true[sorted_indices]

            # Count inversions (positive examples ranked after negative examples)
            pos_ranks = jnp.cumsum(y_true_sorted)
            neg_count_before = jnp.arange(len(y_true_sorted)) + 1 - pos_ranks
            auc_sum = jnp.sum(y_true_sorted * neg_count_before)

            return auc_sum / (n_pos * n_neg)

        return jnp.where((n_pos > 0) & (n_neg > 0), compute_auc(), 0.5)

    def _compute_mrr_impl(self, y_true, y_score):
        """JAX implementation of Mean Reciprocal Rank."""
        # Sort indices in descending order of scores
        order = jnp.argsort(-y_score)
        y_true_sorted = y_true[order]

        # Calculate reciprocal ranks
        ranks = jnp.arange(1, len(y_true_sorted) + 1)
        rr_scores = y_true_sorted / ranks

        # Return MRR
        n_relevant = jnp.sum(y_true)
        return jnp.where(n_relevant > 0, jnp.sum(rr_scores) / n_relevant, 0.0)

    def _dcg_score_impl(self, y_true, y_score, k):
        """JAX implementation of Discounted Cumulative Gain (DCG)@k."""
        k = min(k, len(y_true))

        # Sort by score in descending order and take top k
        order = jnp.argsort(-y_score)
        y_true_sorted = y_true[order[:k]]

        # Calculate DCG
        gains = jnp.power(2.0, y_true_sorted) - 1.0
        discounts = jnp.log2(jnp.arange(2, k + 2))

        return jnp.sum(gains / discounts)

    def _compute_ndcg_impl(self, y_true, y_score, k):
        """JAX implementation of Normalized DCG@k."""
        actual_dcg = self._dcg_score_jax(y_true, y_score, k)
        ideal_dcg = self._dcg_score_jax(y_true, y_true, k)  # Best possible DCG

        return jnp.where(ideal_dcg > 0, actual_dcg / ideal_dcg, 0.0)

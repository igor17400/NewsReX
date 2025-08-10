import keras
import jax
import jax.numpy as jnp
from functools import partial
import logging

# Setup logger
logger = logging.getLogger(__name__)


def _compute_auc_jax(y_true, y_pred):
    """Pure JAX implementation of AUC calculation using trapezoidal rule.
    
    This avoids sklearn dependency and can run on GPU.
    """
    # Handle edge cases
    n_pos = jnp.sum(y_true)
    n_neg = jnp.sum(1 - y_true)

    # If only one class present, return 0.5
    def single_class_auc():
        # Ensure consistent dtype with compute_auc branch
        return jnp.array(0.5, dtype=y_pred.dtype)

    def compute_auc():
        # Sort by predictions in descending order
        sorted_indices = jnp.argsort(-y_pred)
        sorted_y_true = y_true[sorted_indices]

        # Calculate cumulative sums
        tps = jnp.cumsum(sorted_y_true)
        fps = jnp.cumsum(1 - sorted_y_true)

        # Add initial point (0, 0) with consistent dtype
        tps = jnp.concatenate([jnp.array([0], dtype=y_true.dtype), tps])
        fps = jnp.concatenate([jnp.array([0], dtype=y_true.dtype), fps])

        # Calculate TPR and FPR
        tpr = tps / n_pos
        fpr = fps / n_neg

        # Calculate AUC using trapezoidal rule
        # AUC = sum of trapezoid areas
        dx = fpr[1:] - fpr[:-1]
        y_avg = (tpr[1:] + tpr[:-1]) / 2
        auc = jnp.sum(dx * y_avg)

        return auc

    # Use conditional to handle edge cases
    return jax.lax.cond(
        (n_pos == 0) | (n_neg == 0),
        lambda _: single_class_auc(),
        lambda _: compute_auc(),
        None
    )


class NewsRecommenderMetricsOptimized:
    """JAX-optimized metrics for evaluating news recommendation models.
    
    Key optimizations:
    - Pure JAX AUC implementation (no sklearn)
    - Fully JIT-compiled metric functions
    - Vectorized batch processing
    - GPU-accelerated computations
    """

    METRIC_NAMES = ["auc", "mrr", "ndcg@5", "ndcg@10"]

    def __init__(self):
        """Initialize the JAX-optimized metrics calculator."""
        # JIT compile all metric functions for maximum performance
        self._compute_auc_jax = jax.jit(_compute_auc_jax)
        self._compute_mrr_jax = jax.jit(self._compute_mrr_impl)
        self._compute_ndcg_jax = jax.jit(self._compute_ndcg_impl, static_argnames=['k'])
        self._dcg_score_jax = jax.jit(self._dcg_score_impl, static_argnames=['k'])

        # JIT compile the batch processing function
        self._compute_metrics_batch_jax = jax.jit(self._compute_metrics_batch)

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

        # Batch processing using fully vectorized JAX operations
        metrics = self._compute_metrics_batch_jax(y_true, y_pred_logits)

        return {
            "auc": float(metrics[0]),
            "mrr": float(metrics[1]),
            "ndcg@5": float(metrics[2]),
            "ndcg@10": float(metrics[3]),
        }

    def _compute_metrics_batch(self, y_true, y_pred_logits):
        """Fully JIT-compiled batch metrics computation."""
        # Vectorized computation using vmap
        auc_scores = jax.vmap(self._compute_auc_jax)(y_true, y_pred_logits)
        mrr_scores = jax.vmap(self._compute_mrr_jax)(y_true, y_pred_logits)
        ndcg5_scores = jax.vmap(partial(self._compute_ndcg_jax, k=5))(y_true, y_pred_logits)
        ndcg10_scores = jax.vmap(partial(self._compute_ndcg_jax, k=10))(y_true, y_pred_logits)

        # Return mean of all metrics
        return jnp.array([
            jnp.mean(auc_scores),
            jnp.mean(mrr_scores),
            jnp.mean(ndcg5_scores),
            jnp.mean(ndcg10_scores)
        ])

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

    def _compute_mrr_impl(self, y_true, y_score):
        """JAX implementation of Mean Reciprocal Rank.
        
        MRR = sum(relevance_i / rank_i) / sum(relevance_i)
        This follows the MIND dataset evaluation convention.
        """
        # Sort indices in descending order of scores
        order = jnp.argsort(y_score)[::-1]
        y_true_sorted = y_true[order]

        # Calculate reciprocal ranks
        rr_scores = y_true_sorted / (jnp.arange(len(y_true_sorted)) + 1)

        # Return MRR following MIND convention: sum(rr_scores) / sum(y_true)
        return jnp.where(jnp.sum(y_true) > 0, jnp.sum(rr_scores) / jnp.sum(y_true), 0.0)

    def _dcg_score_impl(self, y_true, y_score, k):
        """JAX implementation of Discounted Cumulative Gain (DCG)@k.
        
        DCG@k = sum_{i=1}^k (2^rel_i - 1) / log2(i + 1)
        where rel_i is the relevance of item at rank i.
        """
        # Sort by score in descending order
        order = jnp.argsort(y_score)[::-1]
        y_true_sorted = y_true[order[:k]]

        # Calculate gains: 2^relevance - 1
        gains = jnp.power(2, y_true_sorted) - 1

        # Calculate discounts: log2(rank + 1) where rank starts from 1
        # For position i (0-indexed), the rank is i+1, so discount is log2(i+2)
        discounts = jnp.log2(jnp.arange(len(y_true_sorted)) + 2)

        return jnp.sum(gains / discounts)

    def _compute_ndcg_impl(self, y_true, y_score, k):
        """JAX implementation of Normalized DCG@k."""
        actual_dcg = self._dcg_score_jax(y_true, y_score, k)
        ideal_dcg = self._dcg_score_jax(y_true, y_true, k)  # Best possible DCG

        return jnp.where(ideal_dcg > 0, actual_dcg / ideal_dcg, 0.0)


# Backward compatibility - use optimized version by default
NewsRecommenderMetrics = NewsRecommenderMetricsOptimized

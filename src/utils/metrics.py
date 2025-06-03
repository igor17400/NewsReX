import numpy as np
from sklearn.metrics import roc_auc_score
import logging

# Setup logger
logger = logging.getLogger(__name__)


class NewsRecommenderMetrics:
    """Stateless metrics for evaluating news recommendation models."""

    METRIC_NAMES = ["auc", "mrr", "ndcg@5", "ndcg@10"]

    def __init__(self):
        """Initialize the metrics calculator."""
        pass

    def compute_metrics(self, y_true, y_pred_logits, progress=None):
        """
        Compute metrics for a single impression or batch.

        Args:
            y_true: np.ndarray or tf.Tensor, shape (batch_size, num_candidates) or (num_candidates,)
            y_pred_logits: np.ndarray or tf.Tensor, shape (batch_size, num_candidates) or (num_candidates,)
            progress: Optional progress bar for logging

        Returns:
            dict: Dictionary of metric names and their values
        """
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred_logits = np.asarray(y_pred_logits)

        if y_true.ndim == 1:
            return self._compute_metrics_single(y_true, y_pred_logits)

        results = {k: [] for k in self.METRIC_NAMES}
        for i in range(y_true.shape[0]):
            single = self._compute_metrics_single(y_true[i], y_pred_logits[i])
            for k in self.METRIC_NAMES:
                results[k].append(single[k])
        return {k: float(np.mean(v)) for k, v in results.items()}

    def compute_metrics_from_scores(self, y_true_grouped, y_pred_scores_grouped, progress=None):
        """
        Compute metrics from grouped scores (for compatibility with existing code).

        Args:
            y_true_grouped: tf.Tensor or np.ndarray, shape (batch_size, num_candidates)
            y_pred_scores_grouped: tf.Tensor or np.ndarray, shape (batch_size, num_candidates)
            progress: Optional progress bar for logging

        Returns:
            dict: Dictionary of metric names and their values
        """
        return self.compute_metrics(y_true_grouped, y_pred_scores_grouped, progress)

    def _compute_metrics_single(self, y_true, y_pred_logits):
        """Compute metrics for a single impression."""
        try:
            # AUC
            auc = roc_auc_score(y_true, y_pred_logits)
        except Exception as e:
            logger.warning(f"Error computing AUC: {e}")
            auc = 0.0

        # MRR
        mrr = self._compute_mrr(y_true, y_pred_logits)

        # nDCG
        ndcg5 = self._compute_ndcg(y_true, y_pred_logits, k=5)
        ndcg10 = self._compute_ndcg(y_true, y_pred_logits, k=10)

        return {
            "auc": auc,
            "mrr": mrr,
            "ndcg@5": ndcg5,
            "ndcg@10": ndcg10,
        }

    def _compute_mrr(self, y_true, y_score):
        """Compute Mean Reciprocal Rank."""
        order = np.argsort(y_score)[::-1]  # Sort in descending order
        y_true = np.take(y_true, order)
        rr_score = y_true / (np.arange(len(y_true)) + 1)
        return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0

    def _compute_ndcg(self, y_true, y_score, k):
        """Compute NDCG@k."""
        k = min(k, len(y_true))

        # Compute DCG
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2**y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        dcg = np.sum(gains / discounts)

        # Compute ideal DCG
        ideal_order = np.argsort(y_true)[::-1]
        ideal_y_true = np.take(y_true, ideal_order[:k])
        ideal_gains = 2**ideal_y_true - 1
        ideal_dcg = np.sum(ideal_gains / discounts)

        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

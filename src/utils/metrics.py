import logging
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

# Setup logger
logger = logging.getLogger(__name__)


class NewsRecommenderMetrics:
    """Metrics for evaluating news recommendation models"""

    def __init__(self):
        self.metrics = {
            "auc": tf.keras.metrics.Mean(),
            "mrr": tf.keras.metrics.Mean(),
            "ndcg@5": tf.keras.metrics.Mean(),
            "ndcg@10": tf.keras.metrics.Mean(),
        }

    def compute_metrics(self, labels, predictions, progress=None):
        """Compute metrics for the predictions."""
        # Convert to numpy for metric computation
        labels_np = labels.numpy()
        predictions_np = predictions.numpy()

        # Process each impression in the batch
        batch_size = tf.shape(labels)[0]

        for i in range(batch_size):
            # Get predictions and labels for this sequence
            seq_predictions = predictions_np[i]
            seq_labels = labels_np[i]

            # Compute AUC using sklearn
            auc = roc_auc_score(seq_labels, seq_predictions)
            self.metrics["auc"].update_state(auc)

            # Compute MRR
            mrr_score = self._compute_mrr(seq_labels, seq_predictions)
            self.metrics["mrr"].update_state(mrr_score)

            # Compute NDCG@5 and NDCG@10
            ndcg5_score = self._compute_ndcg(seq_labels, seq_predictions, k=5)
            self.metrics["ndcg@5"].update_state(ndcg5_score)

            ndcg10_score = self._compute_ndcg(seq_labels, seq_predictions, k=10)
            self.metrics["ndcg@10"].update_state(ndcg10_score)

        # Get final results
        results = {
            "auc": self.metrics["auc"].result(),
            "mrr": self.metrics["mrr"].result(),
            "ndcg@5": self.metrics["ndcg@5"].result(),
            "ndcg@10": self.metrics["ndcg@10"].result(),
        }

        # Reset metrics for next batch
        for metric in self.metrics.values():
            metric.reset_state()

        return results

    def _compute_mrr(self, y_true, y_score):
        """Compute Mean Reciprocal Rank using the weighted sum approach."""
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order)
        rr_score = y_true / (np.arange(len(y_true)) + 1)
        return np.sum(rr_score) / np.sum(y_true)

    def _compute_ndcg(self, y_true, y_score, k):
        """Compute NDCG@k using the provided formula."""
        # Ensure k is not larger than sequence length
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

        # Avoid division by zero
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

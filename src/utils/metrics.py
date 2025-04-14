from typing import Dict, Optional
import logging
import tensorflow as tf
from rich.progress import Progress

# Setup logger
logger = logging.getLogger(__name__)


class NewsRecommenderMetrics:
    """Metrics for evaluating news recommendation models"""

    def __init__(self):
        self.metrics = {
            "auc": tf.keras.metrics.AUC(),
            "mrr": tf.keras.metrics.Mean(),
            "ndcg@5": tf.keras.metrics.Mean(),
            "ndcg@10": tf.keras.metrics.Mean(),
        }

    def compute_metrics(self, labels, predictions, masks, progress=None):
        """Compute metrics using only valid positions."""
        # Apply masks to predictions and labels
        masked_predictions = predictions * tf.cast(masks, predictions.dtype)
        masked_labels = labels * tf.cast(masks, labels.dtype)

        # Compute metrics for each batch
        results = {}
        
        # Process AUC first - handle all valid positions at once
        valid_indices = tf.where(masks > 0)
        valid_predictions = tf.gather_nd(masked_predictions, valid_indices)
        valid_labels = tf.gather_nd(masked_labels, valid_indices)
        self.metrics["auc"].update_state(valid_labels, valid_predictions)
        results["auc"] = self.metrics["auc"].result()
        self.metrics["auc"].reset_state()

        # Process other metrics in a vectorized way
        batch_size = tf.shape(labels)[0]
        
        # Get valid positions for each sequence
        valid_positions = tf.cast(masks > 0, tf.int32)
        sequence_lengths = tf.reduce_sum(valid_positions, axis=1)
        
        # Compute MRR and NDCG for each sequence
        for i in range(batch_size):
            seq_length = sequence_lengths[i]
            if seq_length > 0:
                # Get valid predictions and labels for this sequence
                seq_predictions = masked_predictions[i, :seq_length]
                seq_labels = masked_labels[i, :seq_length]
                
                # Compute MRR
                mrr_score = self._compute_mrr(seq_labels, seq_predictions)
                self.metrics["mrr"].update_state(mrr_score)
                
                # Compute NDCG@5 and NDCG@10
                ndcg5_score = self._compute_ndcg(seq_labels, seq_predictions, k=5)
                self.metrics["ndcg@5"].update_state(ndcg5_score)
                
                ndcg10_score = self._compute_ndcg(seq_labels, seq_predictions, k=10)
                self.metrics["ndcg@10"].update_state(ndcg10_score)
                
                if progress is not None:
                    progress_bar, task_id = progress
                    progress_bar.update(task_id, advance=1)

        # Get final results
        results["mrr"] = self.metrics["mrr"].result()
        results["ndcg@5"] = self.metrics["ndcg@5"].result()
        results["ndcg@10"] = self.metrics["ndcg@10"].result()
        
        # Reset metrics for next batch
        self.metrics["mrr"].reset_state()
        self.metrics["ndcg@5"].reset_state()
        self.metrics["ndcg@10"].reset_state()

        return results

    def _compute_mrr(self, labels, predictions):
        """Compute Mean Reciprocal Rank."""
        # Sort predictions and get ranks
        sorted_indices = tf.argsort(predictions, direction="DESCENDING")
        ranks = tf.argsort(sorted_indices) + 1
        
        # Get rank of first positive
        positive_ranks = tf.boolean_mask(ranks, labels > 0)
        if tf.size(positive_ranks) > 0:
            first_positive_rank = tf.reduce_min(positive_ranks)
            return 1.0 / tf.cast(first_positive_rank, tf.float32)
        return 0.0

    def _compute_ndcg(self, labels, predictions, k):
        """Compute NDCG@k."""
        # Ensure k is not larger than sequence length
        k = tf.minimum(k, tf.shape(labels)[0])
        
        # Sort predictions and get top k
        sorted_indices = tf.argsort(predictions, direction="DESCENDING")
        top_k_indices = sorted_indices[:k]
        
        # Get relevance scores for top k
        top_k_labels = tf.gather(labels, top_k_indices)
        
        # Compute DCG
        ranks = tf.range(1, k + 1, dtype=tf.float32)
        dcg = tf.reduce_sum(top_k_labels / tf.math.log1p(ranks))
        
        # Compute ideal DCG
        ideal_sorted = tf.sort(labels, direction="DESCENDING")
        ideal_top_k = ideal_sorted[:k]
        ideal_dcg = tf.reduce_sum(ideal_top_k / tf.math.log1p(ranks))
        
        # Avoid division by zero
        return tf.cond(
            ideal_dcg > 0, lambda: dcg / ideal_dcg, lambda: tf.constant(0.0, dtype=tf.float32)
        )

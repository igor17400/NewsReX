from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf


class NewsRecommenderMetrics:
    """Metrics for evaluating news recommendation models"""

    @staticmethod
    def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUC score"""
        return tf.keras.metrics.AUC()(y_true, y_pred).numpy()

    @staticmethod
    def mrr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Reciprocal Rank"""
        indices = np.argsort(-y_pred)
        ranks = np.where(y_true[indices] == 1)[0] + 1
        return np.mean(1.0 / ranks) if len(ranks) > 0 else 0.0

    @staticmethod
    def ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        indices = np.argsort(-y_pred)[:k]
        dcg = np.sum(y_true[indices] / np.log2(np.arange(2, len(indices) + 2)))

        # Calculate ideal DCG
        ideal_indices = np.argsort(-y_true)[:k]
        idcg = np.sum(y_true[ideal_indices] / np.log2(np.arange(2, len(ideal_indices) + 2)))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def group_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        impression_ids: List[str],
        masks: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate metrics for grouped impressions"""
        unique_impressions = np.unique(impression_ids)
        metrics: Dict[str, List[float]] = {"auc": [], "mrr": [], "ndcg@5": [], "ndcg@10": []}

        for imp_id in unique_impressions:
            mask = impression_ids == imp_id
            if masks is not None:
                # Apply additional mask for padded values
                mask = mask & (masks[mask] == 1)

            if np.sum(y_true[mask]) > 0:  # Only consider impressions with positive samples
                metrics["auc"].append(NewsRecommenderMetrics.auc(y_true[mask], y_pred[mask]))
                metrics["mrr"].append(NewsRecommenderMetrics.mrr(y_true[mask], y_pred[mask]))
                metrics["ndcg@5"].append(NewsRecommenderMetrics.ndcg(y_true[mask], y_pred[mask], k=5))
                metrics["ndcg@10"].append(NewsRecommenderMetrics.ndcg(y_true[mask], y_pred[mask], k=10))

        return {k: np.mean(v) for k, v in metrics.items()}

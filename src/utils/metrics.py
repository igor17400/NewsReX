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
        metrics: Dict[str, List[float]] = {"auc": [], "mrr": [], "ndcg@5": [], "ndcg@10": []}
        
        # Convert tensors to numpy arrays if needed
        y_true = y_true.numpy() if hasattr(y_true, 'numpy') else y_true
        y_pred = y_pred.numpy() if hasattr(y_pred, 'numpy') else y_pred
        masks = masks.numpy() if hasattr(masks, 'numpy') else masks
        
        unique_impressions = np.unique(impression_ids)
        
        for imp_id in unique_impressions:
            # Find indices for this impression ID
            indices = [i for i, x in enumerate(impression_ids) if x == imp_id]
            
            # Get data for this impression
            imp_true = y_true[indices]
            imp_pred = y_pred[indices]
            imp_mask = masks[indices] if masks is not None else None
            
            if imp_mask is not None:
                # Apply mask
                valid_positions = imp_mask == 1
                imp_true = imp_true[valid_positions]
                imp_pred = imp_pred[valid_positions]
            
            if np.sum(imp_true) > 0:  # Only consider impressions with positive samples
                metrics["auc"].append(NewsRecommenderMetrics.auc(imp_true, imp_pred))
                metrics["mrr"].append(NewsRecommenderMetrics.mrr(imp_true, imp_pred))
                metrics["ndcg@5"].append(NewsRecommenderMetrics.ndcg(imp_true, imp_pred, k=5))
                metrics["ndcg@10"].append(NewsRecommenderMetrics.ndcg(imp_true, imp_pred, k=10))

        return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}

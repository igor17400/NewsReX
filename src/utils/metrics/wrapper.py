import keras
import numpy as np

from src.utils.metrics.functions_optimized import NewsRecommenderMetricsOptimized as NewsRecommenderMetrics


class NewsRecommenderKerasMetric(keras.metrics.Metric):
    """Keras 3 compatible wrapper for news recommendation metrics."""
    
    def __init__(self, metric_name: str, custom_metrics_engine: NewsRecommenderMetrics, **kwargs):
        super().__init__(name=metric_name, **kwargs)
        self.metric_name = metric_name
        self.custom_metrics_engine = custom_metrics_engine
        
        # State variables to accumulate predictions and labels
        self.total_predictions = self.add_weight(name="total_predictions", initializer="zeros", shape=())
        self.total_labels = self.add_weight(name="total_labels", initializer="zeros", shape=())
        
        # Lists to store batch data (note: this is not ideal for very large datasets)
        self.predictions_list = []
        self.labels_list = []
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state with batch predictions and labels."""
        # Convert to numpy for processing
        y_true_np = keras.ops.convert_to_numpy(y_true)
        y_pred_np = keras.ops.convert_to_numpy(y_pred)
        
        # Store batch data
        self.predictions_list.append(y_pred_np)
        self.labels_list.append(y_true_np)
        
        # Update counters
        batch_size = keras.ops.shape(y_true)[0]
        self.total_predictions.assign_add(keras.ops.cast(batch_size, self.dtype))
        
    def result(self):
        """Compute the final metric value."""
        if not self.predictions_list or not self.labels_list:
            return 0.0
            
        # Concatenate all batch data
        all_predictions = np.concatenate(self.predictions_list, axis=0)
        all_labels = np.concatenate(self.labels_list, axis=0)
        
        # Compute custom metrics
        try:
            metrics_dict = self.custom_metrics_engine.compute_metrics(
                y_true=all_labels,
                y_pred_logits=all_predictions,
                progress=None  # No progress bar in metric computation
            )
            
            # Return the specific metric
            return float(metrics_dict.get(self.metric_name, 0.0))
            
        except Exception as e:
            print(f"Error computing {self.metric_name}: {e}")
            return 0.0
    
    def reset_state(self):
        """Reset the metric state."""
        self.total_predictions.assign(0.0)
        self.total_labels.assign(0.0)
        self.predictions_list.clear()
        self.labels_list.clear()


class AUCNewsMetric(NewsRecommenderKerasMetric):
    """Keras 3 compatible AUC metric for news recommendation."""
    
    def __init__(self, custom_metrics_engine: NewsRecommenderMetrics, **kwargs):
        super().__init__(metric_name="auc", custom_metrics_engine=custom_metrics_engine, **kwargs)


class MRRNewsMetric(NewsRecommenderKerasMetric):
    """Keras 3 compatible MRR metric for news recommendation."""
    
    def __init__(self, custom_metrics_engine: NewsRecommenderMetrics, **kwargs):
        super().__init__(metric_name="mrr", custom_metrics_engine=custom_metrics_engine, **kwargs)


class NDCG5NewsMetric(NewsRecommenderKerasMetric):
    """Keras 3 compatible nDCG@5 metric for news recommendation."""
    
    def __init__(self, custom_metrics_engine: NewsRecommenderMetrics, **kwargs):
        super().__init__(metric_name="ndcg@5", custom_metrics_engine=custom_metrics_engine, **kwargs)


class NDCG10NewsMetric(NewsRecommenderKerasMetric):
    """Keras 3 compatible nDCG@10 metric for news recommendation."""
    
    def __init__(self, custom_metrics_engine: NewsRecommenderMetrics, **kwargs):
        super().__init__(metric_name="ndcg@10", custom_metrics_engine=custom_metrics_engine, **kwargs)


def create_news_metrics(custom_metrics_engine: NewsRecommenderMetrics) -> list:
    """Create a list of Keras 3 compatible news recommendation metrics.
    
    Args:
        custom_metrics_engine: The custom metrics calculator
        
    Returns:
        List of Keras metrics that can be used in model.compile()
    """
    return [
        AUCNewsMetric(custom_metrics_engine),
        MRRNewsMetric(custom_metrics_engine),
        NDCG5NewsMetric(custom_metrics_engine),
        NDCG10NewsMetric(custom_metrics_engine),
    ]


class LightweightNewsMetrics:
    """Lightweight version that only computes metrics on validation, not during training."""
    
    @staticmethod
    def create_training_metrics() -> list:
        """Create lightweight metrics for training monitoring.
        
        These are standard Keras metrics that provide quick feedback during training
        without the computational overhead of custom news recommendation metrics.
        """
        return [
            'accuracy',  # Standard accuracy for quick monitoring
            keras.metrics.AUC(name='keras_auc'),  # Standard AUC
        ]
    
    @staticmethod
    def should_use_lightweight_metrics(cfg) -> bool:
        """Determine if lightweight metrics should be used during training.
        
        Args:
            cfg: Configuration object
            
        Returns:
            True if lightweight metrics should be used, False otherwise
        """
        # Use lightweight metrics if fast evaluation is enabled
        # (custom metrics will be computed in callbacks)
        return cfg.eval.fast_evaluation
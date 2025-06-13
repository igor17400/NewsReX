from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm

from utils.io import save_predictions_to_file_fn
from datasets.dataloader import NewsBatchDataloader, UserHistoryBatchDataloader


class BaseModel(Model):
    """Base class for news recommendation models.

    This class provides common functionality for news recommendation models,
    including methods for fast evaluation using precomputed vectors.
    """

    def precompute_news_vectors(
        self,
        news_dataloader: NewsBatchDataloader,
        progress: Any,
    ) -> Dict[str, np.ndarray]:
        """Precompute vectors for all news articles.

        Args:
            news_dataloader: NewsBatchDataloader instance for news articles
            progress: Progress bar manager

        Returns:
            Dictionary mapping news IDs to their vectors
        """
        news_vecs_dict = {}
        total_news = len(news_dataloader)

        news_progress = progress.add_task(
            "Computing news vectors...", total=total_news, visible=True
        )

        for batch in news_dataloader:
            # Convert batch to numpy arrays
            news_ids = batch["news_id"]
            news_features = batch["news_features"]

            batch_vecs = self.newsencoder(news_features, training=False).numpy()

            for i, news_id in enumerate(news_ids):
                news_vecs_dict[str(news_id)] = batch_vecs[i]

            progress.update(news_progress, advance=len(news_ids))

        progress.remove_task(news_progress)
        return news_vecs_dict

    def precompute_user_vectors(
        self, user_dataloader: UserHistoryBatchDataloader, progress: Optional[tqdm] = None
    ) -> Dict[int, np.ndarray]:
        """Pre-compute user vectors for fast evaluation.

        Args:
            user_dataloader: UserHistoryBatchDataloader for user history data
            progress: Optional progress bar

        Returns:
            Dictionary mapping impression IDs based on the users behaviors to their vectors
        """
        user_vecs_dict = {}
        user_progress = progress.add_task(
            "Computing user vectors...", total=len(user_dataloader), visible=True
        )

        for impression_ids, features in user_dataloader:
            # Get user representation from history
            user_vec = self.userencoder(features, training=False)

            # Store user vector for each impression in the batch
            for i, imp_id in enumerate(impression_ids):
                user_vecs_dict[int(imp_id)] = user_vec[i].numpy()

            progress.update(user_progress, advance=len(impression_ids))

        progress.remove_task(user_progress)

        return user_vecs_dict

    def fast_evaluate(
        self,
        user_hist_dataloader,
        news_dataloader,
        impression_iterator,
        metrics_calculator,
        progress,
        mode="validate",
        save_predictions_path=None,
        epoch=None,
    ) -> Dict[str, float]:
        """Fast evaluation of the model using precomputed vectors and dataloader iterators."""
        # 1. Precompute news vectors
        news_vecs_dict = self.precompute_news_vectors(news_dataloader, progress)

        # 2. Precompute user vectors
        user_vecs_dict = self.precompute_user_vectors(user_hist_dataloader, progress)

        # 3. Score impressions using precomputed user and news vectors
        group_labels_list = []
        group_preds_list = []
        predictions_to_save = {}

        # Create progress bar for impressions
        impression_progress = progress.add_task(
            "Processing impressions...", total=len(impression_iterator), visible=True
        )

        for impression in impression_iterator:
            _, labels, impression_id, cand_ids = impression

            # Get user vector for this impression
            user_vector = user_vecs_dict[impression_id.numpy()[0]]

            # Get news vectors for candidate news
            cand_ids_np = cand_ids.numpy()[0]  # Get numpy array of candidate IDs
            news_vectors = []
            for nid in cand_ids_np:
                news_key = f"N{str(nid)}"
                news_vectors.append(news_vecs_dict[news_key])
            
            # Calculate scores using dot product
            scores = np.dot(np.stack(news_vectors, axis=0), user_vector)

            group_labels_list.append(labels.numpy())
            group_preds_list.append(scores)

            if save_predictions_path:
                predictions_to_save[str(cand_ids.numpy())] = (
                    labels.numpy().tolist(),
                    scores.tolist(),
                )

            progress.update(impression_progress, advance=1)

        progress.remove_task(impression_progress)

        # 4. Compute metrics
        final_metrics = self._compute_metrics(
            group_labels_list, group_preds_list, metrics_calculator, progress
        )
        final_metrics["num_impressions"] = len(group_labels_list)

        if save_predictions_path:
            save_predictions_to_file_fn(predictions_to_save, save_predictions_path, epoch, mode)

        return final_metrics

    def _compute_metrics(
        self,
        group_labels_list: List[np.ndarray],
        group_preds_list: List[np.ndarray],
        metrics_calculator: Any,
        progress: Any,
    ) -> Dict[str, float]:
        """Computes final metrics from collected predictions and labels."""
        val_loss_total = 0.0
        num_valid_impressions_for_loss = 0
        metric_values_agg = {key: [] for key in metrics_calculator.METRIC_NAMES}

        for labels_np, scores_np in zip(group_labels_list, group_preds_list):
            if labels_np.size == 0 or scores_np.size == 0:
                continue

            labels_tf = tf.constant([labels_np], dtype=tf.float32)
            scores_tf_logits = tf.constant([scores_np], dtype=tf.float32)

            if self._has_nan_or_inf(scores_tf_logits, labels_tf):
                progress.console.print("[WARNING fast_evaluate] NaN/Inf detected. Skipping loss.")
                continue

            scores_tf_probs = tf.nn.softmax(scores_tf_logits, axis=-1)

            try:
                loss = self.compute_loss(y=labels_tf, y_pred=scores_tf_probs, training=False)
                if loss is not None:
                    val_loss_total += loss.numpy()
                    num_valid_impressions_for_loss += 1
            except Exception as e:
                progress.console.print(f"[WARNING fast_evaluate] Error calculating loss: {e}.")

            impression_metrics = metrics_calculator.compute_metrics(
                y_true=labels_tf, y_pred_logits=scores_tf_logits
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
            final_metrics[metric_name] = np.mean(values_list) if values_list else 0.0

        return final_metrics

    def _has_nan_or_inf(self, *tensors: tf.Tensor) -> bool:
        """Checks if any tensor in the list contains NaN or Inf values."""
        for tensor in tensors:
            if tf.reduce_any(tf.math.is_nan(tensor)) or tf.reduce_any(tf.math.is_inf(tensor)):
                return True
        return False

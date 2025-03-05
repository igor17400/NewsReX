import tensorflow as tf
from typing import Dict, Any

class NewsDataLoader:
    """Generic dataloader for news recommendation datasets."""
    
    @staticmethod
    def create_train_dataset(
        history_tokens: tf.Tensor,
        impression_tokens: tf.Tensor,
        labels: tf.Tensor,
        histories_masks: tf.Tensor,
        batch_size: int,
        buffer_size: int = 10000
    ) -> tf.data.Dataset:
        """Create training dataset with both history and impression masks."""
        features = {
            "user_tokens": history_tokens,
            "user_masks": histories_masks,
            "cand_tokens": impression_tokens,
        }
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def create_eval_dataset(
        history_tokens: tf.Tensor,
        impression_tokens: tf.Tensor,
        labels: tf.Tensor,
        histories_masks: tf.Tensor,
        batch_size: int,
    ) -> tf.data.Dataset:
        """Create evaluation dataset with token inputs."""
        features = {
            "user_tokens": history_tokens,
            "user_masks": histories_masks,
            "cand_tokens": impression_tokens,
        }
        
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def prepare_features(
        history_embeddings: tf.Tensor,
        impression_embeddings: tf.Tensor
    ) -> Dict[str, Dict[str, tf.Tensor]]:
        """Prepare features dictionary for the model."""
        return {
            "impression_news": {"title": impression_embeddings},
            "history_news": {"title": history_embeddings}
        } 
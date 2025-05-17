import tensorflow as tf
from typing import Dict, Any, Iterator, Tuple, List
import numpy as np


class ImpressionIterator:
    """Iterator for processing impressions during validation/testing."""

    def __init__(
        self,
        history_tokens: List[List[int]],
        impression_tokens: List[List[List[int]]],
        labels: List[List[float]],
        impression_ids: List[int],
    ):
        """Initialize iterator with data references.
        
        Args:
            history_tokens: List of user history token sequences
            impression_tokens: List of impression token sequences (variable length)
            labels: List of label sequences (variable length)
            impression_ids: List of impression IDs
        """
        self.history_tokens = history_tokens
        self.impression_tokens = impression_tokens
        self.labels = labels
        self.impression_ids = impression_ids
        self.num_impressions = len(history_tokens)

    def __iter__(self):
        """Create generator for impressions. Process one impression at a time."""
        for idx in range(self.num_impressions):
            # Get data for this impression
            history = self.history_tokens[idx]
            impressions = self.impression_tokens[idx]
            label = self.labels[idx]
            imp_id = self.impression_ids[idx]

            # Convert single impression to tensors
            features = {
                "hist_tokens": tf.constant([history], dtype=tf.int32),  # Add batch dim
                "cand_tokens": tf.constant([impressions], dtype=tf.int32),  # Add batch dim
            }
            labels = tf.constant([label], dtype=tf.float32)  # Add batch dim
            impression_ids = tf.constant([imp_id], dtype=tf.int32)

            yield features, labels, impression_ids


class NewsDataLoader:
    """Generic dataloader for news recommendation datasets."""

    @staticmethod
    def create_train_dataset(
        history_news_tokens: tf.Tensor,
        candidate_news_tokens: tf.Tensor,
        labels: tf.Tensor,
        batch_size: int,
        buffer_size: int = 10000,
    ) -> tf.data.Dataset:
        """Create training dataset with fixed-length sequences.
        
        This is used for training where we have:
        - Fixed-length sequences from sampling
        - Balanced positive/negative examples
        """
        features = {
            "hist_tokens": history_news_tokens,
            "cand_tokens": candidate_news_tokens,
        }
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    @staticmethod
    def create_eval_iterator(
        history_news_tokens: List[List[int]],
        candidate_news_tokens: List[List[List[int]]],
        labels: List[List[float]],
        impression_ids: List[int],
    ) -> ImpressionIterator:
        """Create evaluation iterator for impression-by-impression evaluation.
        
        For validation/testing, we process one impression at a time since each impression
        is an independent ranking task.
        """
        return ImpressionIterator(
            history_tokens=history_news_tokens,
            impression_tokens=candidate_news_tokens,
            labels=labels,
            impression_ids=impression_ids,
        )


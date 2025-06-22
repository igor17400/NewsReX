import tensorflow as tf
from typing import List
import numpy as np


class ImpressionIterator:
    """Iterator for processing impressions one at a time during validation/testing."""

    def __init__(
        self,
        impression_tokens: List[List[List[int]]],
        impression_abstract_tokens: List[List[int]],
        impression_category: List[List[int]],
        impression_subcategory: List[List[int]],
        labels: List[List[float]],
        impression_ids: List[int],
        candidate_ids: List[int],
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
    ):
        """Initialize iterator with impression data.

        Args:
            impression_tokens: List of impression token sequences (variable length)
            impression_abstract_tokens: List of impression abstract token sequences
            impression_category: List of impression category sequences
            impression_subcategory: List of impression subcategory sequences
            labels: List of label sequences (variable length)
            impression_ids: List of impression IDs
            candidate_ids: List of candidate news IDs
        """
        self.impression_tokens = impression_tokens
        self.impression_abstract_tokens = impression_abstract_tokens
        self.impression_category = impression_category
        self.impression_subcategory = impression_subcategory
        self.labels = labels
        self.impression_ids = impression_ids
        self.candidate_ids = candidate_ids
        self.num_impressions = len(labels)
        policy = tf.keras.mixed_precision.global_policy()
        self.float_dtype = policy.compute_dtype

        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self):
        """Create generator for processing one impression at a time."""
        for idx in range(self.num_impressions):
            # Get data for this impression
            impression_features = []

            if self.process_title:
                # Title is already 2D (list of arrays)
                impression_features.append(np.array(self.impression_tokens[idx]))
            if self.process_abstract:
                # Abstract is already 2D (list of arrays)
                impression_features.append(np.array(self.impression_abstract_tokens[idx]))
            if self.process_category:
                # Convert category to 2D array and transpose to match title/abstract shape
                category = np.array(self.impression_category[idx])
                category = np.expand_dims(category, axis=0)  # Make it 2D
                category = np.transpose(category)  # Transpose to match title/abstract shape
                impression_features.append(category)
            if self.process_subcategory:
                # Convert subcategory to 2D array and transpose to match title/abstract shape
                subcategory = np.array(self.impression_subcategory[idx])
                subcategory = np.expand_dims(subcategory, axis=0)  # Make it 2D
                subcategory = np.transpose(subcategory)  # Transpose to match title/abstract shape
                impression_features.append(subcategory)

            # Convert to tensors and concatenate
            features = {
                "impression_features": tf.constant(
                    tf.concat(impression_features, axis=-1), dtype=tf.int32
                )
            }
            labels = tf.constant(self.labels[idx], dtype=self.float_dtype)
            impression_id = tf.constant([self.impression_ids[idx]], dtype=tf.int32)
            candidate_ids = tf.constant([self.candidate_ids[idx]], dtype=tf.int32)

            yield features, labels, impression_id, candidate_ids

    def __len__(self):
        """Return the total number of impressions."""
        return self.num_impressions


class NewsDataLoader:
    """Generic dataloader for news recommendation datasets."""

    @staticmethod
    def create_train_dataset(
        history_news_tokens: tf.Tensor,
        history_news_abstract_tokens: tf.Tensor,
        history_news_category: tf.Tensor,
        history_news_subcategory: tf.Tensor,
        candidate_news_tokens: tf.Tensor,
        candidate_news_abstract_tokens: tf.Tensor,
        candidate_news_category: tf.Tensor,
        candidate_news_subcategory: tf.Tensor,
        labels: tf.Tensor,
        batch_size: int,
        buffer_size: int = 10000,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
    ) -> tf.data.Dataset:
        """Create training dataset with fixed-length sequences."""
        features = {}
        if process_title:
            features["hist_tokens"] = history_news_tokens
            features["cand_tokens"] = candidate_news_tokens
        if process_abstract:
            features["hist_abstract_tokens"] = history_news_abstract_tokens
            features["cand_abstract_tokens"] = candidate_news_abstract_tokens
        if process_category:
            features["hist_category"] = history_news_category
            features["cand_category"] = candidate_news_category
        if process_subcategory:
            features["hist_subcategory"] = history_news_subcategory
            features["cand_subcategory"] = candidate_news_subcategory

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


class NewsBatchDataloader:
    """Dataloader for processing news articles in batches."""

    def __init__(
        self,
        news_ids: tf.Tensor,
        news_tokens: tf.Tensor,
        news_abstract_tokens: tf.Tensor,
        news_category_indices: tf.Tensor,
        news_subcategory_indices: tf.Tensor,
        batch_size: int = 1024,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
    ):
        """Initialize with news data.

        Args:
            news_ids: Tensor of news IDs
            news_tokens: Tensor of news title tokens
            news_abstract_tokens: Tensor of news abstract tokens
            news_category_indices: Tensor of news category indices
            news_subcategory_indices: Tensor of news subcategory indices
            batch_size: Batch size for processing
        """
        self.news_ids = news_ids
        self.news_tokens = news_tokens
        self.news_abstract_tokens = news_abstract_tokens
        self.news_category_indices = news_category_indices
        self.news_subcategory_indices = news_subcategory_indices
        self.batch_size = batch_size
        self.num_news = len(news_ids)

        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self):
        """Create generator for batched news processing."""
        for i in range(0, self.num_news, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_news)

            # Get batch data
            batch_ids = self.news_ids[i:end_idx]
            batch_tokens = self.news_tokens[i:end_idx]
            batch_abstract = self.news_abstract_tokens[i:end_idx]
            batch_category = self.news_category_indices[i:end_idx]
            batch_subcategory = self.news_subcategory_indices[i:end_idx]

            # Ensure all tensors have the same rank by expanding dimensions if needed
            if len(batch_category.shape) == 1:
                batch_category = tf.expand_dims(batch_category, axis=1)
            if len(batch_subcategory.shape) == 1:
                batch_subcategory = tf.expand_dims(batch_subcategory, axis=1)

            batch_features = []
            if self.process_title:
                batch_features.append(batch_tokens)
            if self.process_abstract:
                batch_features.append(batch_abstract)
            if self.process_category:
                batch_features.append(batch_category)
            if self.process_subcategory:
                batch_features.append(batch_subcategory)

            # Concatenate features
            batch_features = tf.concat(batch_features, axis=1)

            yield {"news_id": batch_ids, "news_features": batch_features}

    def __len__(self):
        """Return the total number of news articles."""
        return self.num_news


class UserHistoryBatchDataloader:
    """Dataloader for processing user histories in batches."""

    def __init__(
        self,
        history_tokens: List[List[int]],
        history_abstract_tokens: List[List[int]],
        history_category: List[List[int]],
        history_subcategory: List[List[int]],
        impression_ids: List[int],
        batch_size: int = 32,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
    ):
        """Initialize with user history data.

        Args:
            history_tokens: List of user history token sequences
            history_abstract_tokens: List of user history abstract token sequences
            history_category: List of user history category sequences
            history_subcategory: List of user history subcategory sequences
            impression_ids: List of impression IDs
            batch_size: Batch size for processing
        """
        self.history_tokens = history_tokens
        self.history_abstract_tokens = history_abstract_tokens
        self.history_category = history_category
        self.history_subcategory = history_subcategory
        self.impression_ids = impression_ids
        self.batch_size = batch_size
        self.num_users = len(impression_ids)

        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self):
        """Create generator for batched user history processing."""
        for i in range(0, self.num_users, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_users)

            # Get batch data
            batch_impression_ids = self.impression_ids[i:end_idx]
            batch_history_features = []

            if self.process_title:
                batch_history_features.append(tf.convert_to_tensor(self.history_tokens[i:end_idx]))
            if self.process_abstract:
                batch_history_features.append(
                    tf.convert_to_tensor(self.history_abstract_tokens[i:end_idx])
                )
            if self.process_category:
                category_features = tf.convert_to_tensor(self.history_category[i:end_idx])
                category_features = tf.expand_dims(category_features, axis=-1)
                batch_history_features.append(category_features)
            if self.process_subcategory:
                subcategory_features = tf.convert_to_tensor(self.history_subcategory[i:end_idx])
                subcategory_features = tf.expand_dims(subcategory_features, axis=-1)
                batch_history_features.append(subcategory_features)

            # Concatenate features
            batch_features = tf.concat(batch_history_features, axis=-1)

            yield batch_impression_ids, batch_features

    def __len__(self):
        """Return the total number of users."""
        return self.num_users

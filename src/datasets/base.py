import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd


class BaseNewsDataset(ABC):
    """Abstract base class for news recommendation datasets"""

    def __init__(self) -> None:
        self.n_behaviors = 0  # Track number of behaviors
        self.news_data: Dict[str, np.ndarray] = {}  # Store news features
        self.behaviors_data: Dict[str, np.ndarray] = {}  # Store behaviors data
        self.val_behaviors_data: Dict[str, np.ndarray] = {}  # Store validation behaviors data
        self.root_dir = Path(".")  # Define root directory

    @abstractmethod
    def download_dataset(self) -> None:
        """Download and extract dataset files"""
        pass

    @abstractmethod
    def build_vocabulary(self) -> None:
        """Build vocabulary from training data"""
        pass

    @abstractmethod
    def process_news(self, news_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process news articles into numerical format"""
        pass

    @abstractmethod
    def process_behaviors(self, behaviors_df: pd.DataFrame, news_dict: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Process user behaviors into numerical format"""
        pass

    @abstractmethod
    def get_train_val_data(self) -> Tuple[Tuple, Tuple]:
        """Get processed training and validation data

        Returns:
            Tuple containing:
            - (news_data, train_behaviors_data)
            - (news_data, val_behaviors_data)
        """
        pass

    @abstractmethod
    def get_test_data(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Get processed test data

        Returns:
            Tuple containing:
            - test_news_data: Dictionary of news features
            - test_behaviors_data: Dictionary of behaviors data
        """
        pass

    def download_and_extract(self, url: str, split: str) -> None:
        """Helper method to download and extract dataset files"""
        zip_path = self.root_dir / f"{split}.zip"
        extract_path = self.root_dir / split

        if not extract_path.exists():
            print(f"Downloading {split} set...")
            urllib.request.urlretrieve(url, zip_path)

            print(f"Extracting {split} set...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            zip_path.unlink()


class BaseDataset:
    """Base class for all datasets."""

    def __init__(self) -> None:
        self.n_behaviors = 0  # Track number of behaviors
        self.news_data: Dict[str, np.ndarray] = {}  # Store news features
        self.histories: np.ndarray = np.array([])  # Store histories
        self.labels: np.ndarray = np.array([])  # Store labels
        self.masks: np.ndarray = np.array([])  # Store masks
        self.val_news_data: Dict[str, np.ndarray] = {}  # Store validation news features
        self.val_histories: np.ndarray = np.array([])  # Store validation histories
        self.val_labels: np.ndarray = np.array([])  # Store validation labels
        self.val_masks: np.ndarray = np.array([])  # Store validation masks
        self.test_news_data: Dict[str, np.ndarray] = {}  # Store test news features
        self.test_histories: np.ndarray = np.array([])  # Store test histories
        self.test_labels: np.ndarray = np.array([])  # Store test labels
        self.test_masks: np.ndarray = np.array([])  # Store test masks

    def train_dataloader(self, batch_size: int) -> Iterator[Tuple]:
        """Create batches of training data.

        Args:
            batch_size: Number of samples per batch

        Yields:
            Tuple containing (news_input, history_input, labels, masks) for each batch
        """
        return self._create_dataloader(self.news_data, self.histories, self.labels, self.masks, batch_size, shuffle=True)

    def val_dataloader(self, batch_size: int) -> Iterator[Tuple]:
        """Create batches of validation data."""
        return self._create_dataloader(
            self.val_news_data,
            self.val_histories,
            self.val_labels,
            self.val_masks,
            batch_size,
            shuffle=False,
        )

    def test_dataloader(self, batch_size: int) -> Iterator[Tuple]:
        """Create batches of test data."""
        return self._create_dataloader(
            self.test_news_data,
            self.test_histories,
            self.test_labels,
            self.test_masks,
            batch_size,
            shuffle=False,
        )

    def _create_dataloader(
        self,
        news_data: Dict[str, np.ndarray],
        histories: np.ndarray,
        labels: np.ndarray,
        masks: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
    ) -> Iterator[Tuple]:
        """Generic data loader implementation.

        Args:
            news_data: Dictionary of news features
            histories: User history data
            labels: Click labels
            masks: Masks for valid positions
            batch_size: Batch size
            shuffle: Whether to shuffle the data

        Yields:
            Tuple of (news_input, history_input, labels, masks) for each batch
        """
        # Use number of behaviors as number of samples
        n_samples = self.n_behaviors

        # Create indices and optionally shuffle them
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        # Create batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Get batch data
            news_input_batch = {k: v[batch_indices] for k, v in news_data.items()}
            history_batch = histories[batch_indices]
            labels_batch = labels[batch_indices]
            masks_batch = masks[batch_indices]

            yield news_input_batch, history_batch, labels_batch, masks_batch

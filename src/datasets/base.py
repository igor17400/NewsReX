import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


class BaseNewsDataset(ABC):
    """Abstract base class for news recommendation datasets"""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.root_dir = Path(cfg["dataset"]["root_dir"])
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if "dataset" not in cfg:
            raise AttributeError("Configuration is missing 'dataset' attribute")

        self.dataset_config = cfg["dataset"]

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
    def process_behaviors(
        self, behaviors_df: pd.DataFrame, news_dict: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process user behaviors into numerical format"""
        pass

    @abstractmethod
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get processed training data"""
        pass

    @abstractmethod
    def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get processed validation data"""
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

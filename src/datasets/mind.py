import logging
import pickle
import urllib.request
import zipfile
import os
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer

from datasets.base import BaseNewsDataset
from utils.cache_manager import CacheManager
from utils.embeddings import EmbeddingsManager
from utils.logging import setup_logging
from utils.sampling import ImpressionSampler

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
console = Console()


class MINDDataset(BaseNewsDataset):
    def __init__(
        self,
        name: str,
        version: str,
        urls: Dict,
        max_title_length: int,
        max_abstract_length: int,
        max_history_length: int,
        max_impressions_length: int,
        random_seed: int,
        embedding_type: str = "glove",
        embedding_size: int = 300,
        sampling: DictConfig = None,
        data_fraction: float = 0.001,
        mode: str = "train",
    ):
        super().__init__()  # Initialize base class with no arguments
        self.name = name
        self.version = version
        self.cache_manager = CacheManager()
        self.dataset_path = self.cache_manager.get_dataset_path("mind", version)
        self.urls = urls[version]
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length
        self.max_history_length = max_history_length
        self.max_impressions_length = max_impressions_length
        self.embedding_type = embedding_type
        self.embedding_size = embedding_size
        self.embeddings_manager = EmbeddingsManager(self.cache_manager)
        self.sampler = ImpressionSampler(sampling) if sampling else None
        self.data_fraction = data_fraction
        self.mode = mode

        logger.info(f"Initializing MIND dataset ({version} version)")
        logger.info(f"Data will be stored in: {self.dataset_path}")

        # Create directories if they don't exist
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # Datasets
        self.train_val_news_data: Dict[str, np.ndarray] = {}
        self.train_behaviors_data: Dict[str, np.ndarray] = {}
        self.val_behaviors_data: Dict[str, np.ndarray] = {}
        self.test_news_data: Dict[str, np.ndarray] = {}
        self.test_behaviors_data: Dict[str, np.ndarray] = {}

        # Set random seed
        np.random.seed(random_seed)

        # Load and process data
        self._load_data(mode)

    @property
    def n_train_behaviors(self) -> int:
        """Number of training behaviors"""
        return (
            len(self.train_behaviors_data["labels"]) if hasattr(self, "train_behaviors_data") else 0
        )

    @property
    def n_val_behaviors(self) -> int:
        """Number of validation behaviors"""
        return len(self.val_behaviors_data["labels"]) if hasattr(self, "val_behaviors_data") else 0

    @property
    def n_test_behaviors(self) -> int:
        """Number of test behaviors"""
        return (
            len(self.test_behaviors_data["labels"]) if hasattr(self, "test_behaviors_data") else 0
        )

    def _load_data(self, mode: str = "train") -> None:
        """Load data based on mode.

        Args:
            mode: Either 'train' for training/validation data or 'test' for test data
        """
        preprocessed_path = self.dataset_path / "preprocessed"
        files_exist = (
            (preprocessed_path / "preprocess_news_train_val.pkl").exists()
            and (preprocessed_path / "preprocess_train_val.pkl").exists()
            and (preprocessed_path / "preprocess_news_test.pkl").exists()
            and (preprocessed_path / "preprocess_test.pkl").exists()
        )

        # Load embeddings only if preprocessed files do not exist
        if not files_exist:
            if self.embedding_type == "glove":
                self.embeddings = self.embeddings_manager.load_glove()
                logger.info("GloVe embeddings loaded successfully")
            elif self.embedding_type == "bert":
                self.bert_model, self.bert_tokenizer = self.embeddings_manager.load_bert()
                logger.info("BERT model loaded successfully")

        if not files_exist:
            self._preprocess_data()

        if mode == "train":
            # Load train/val data
            try:
                with open(preprocessed_path / "preprocess_news_train_val.pkl", "rb") as f:
                    self.train_val_news_data = pickle.load(f)
                with open(preprocessed_path / "preprocess_train_val.pkl", "rb") as f:
                    behaviors = pickle.load(f)
                    self.train_behaviors_data = behaviors["train"]
                    self.val_behaviors_data = behaviors["val"]
                logger.info("Loaded preprocessed train/val data")
            except (FileNotFoundError, pickle.UnpicklingError) as e:
                raise RuntimeError("Error loading preprocessed data.") from e

        elif mode == "test":
            # Load only test data
            files_exist = (preprocessed_path / "preprocess_news_test.pkl").exists() and (
                preprocessed_path / "preprocess_test.pkl"
            ).exists()

            if not files_exist:
                self._preprocess_data(mode="test")

            # Load test data
            try:
                with open(preprocessed_path / "preprocess_news_test.pkl", "rb") as f:
                    self.test_news_data = pickle.load(f)
                with open(preprocessed_path / "preprocess_test.pkl", "rb") as f:
                    behaviors = pickle.load(f)
                    self.test_behaviors_data = behaviors["test"]
                logger.info("Loaded preprocessed test data")
            except (FileNotFoundError, pickle.UnpicklingError) as e:
                raise RuntimeError("Error loading preprocessed test data.") from e

        if self.data_fraction < 1.0:
            self._apply_data_fraction(mode)

        self._display_statistics(mode)

    def _preprocess_data(self, mode: str = "train") -> None:
        preprocessed_path = self.dataset_path / "preprocessed"
        # Create preprocessed directory
        os.makedirs(preprocessed_path, exist_ok=True)

        # ----- Process train and validation data
        logger.info("Processing train and validation data...")
        self.train_val_news_data, self.train_behaviors_data, self.val_behaviors_data = (
            self.get_train_val_data()
        )
        # Save train/val data
        logger.info("Saving train/val data...")
        with open(preprocessed_path / "preprocess_news_train_val.pkl", "wb") as f:
            pickle.dump(self.train_val_news_data, f)
        with open(preprocessed_path / "preprocess_train_val.pkl", "wb") as f:
            pickle.dump({"train": self.train_behaviors_data, "val": self.val_behaviors_data}, f)

        # Clear memory
        del self.train_val_news_data
        del self.train_behaviors_data
        del self.val_behaviors_data

        # ----- Process testing data
        logger.info("Processing test data...")
        self.test_news_data, self.test_behaviors_data = self.get_test_data()

        # Save test data
        logger.info("Saving test data...")
        with open(preprocessed_path / "preprocess_news_test.pkl", "wb") as f:
            pickle.dump(self.test_news_data, f)
        with open(preprocessed_path / "preprocess_test.pkl", "wb") as f:
            pickle.dump({"test": self.test_behaviors_data}, f)

        # Clear memory
        del self.test_news_data
        del self.test_behaviors_data

        # Preprocessing complete
        logger.info("Preprocessing complete!")

    def download_dataset(self) -> None:
        """Download and extract MIND dataset if not already present"""
        if self.cache_manager.is_dataset_cached("mind", self.version):
            logger.info(f"Found cached dataset at {self.dataset_path}")
            return

        self.dataset_path.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            for split, url in self.urls.items():
                zip_path = self.dataset_path / f"{split}.zip"
                extract_path = self.dataset_path / split

                if not extract_path.exists():
                    # Download file
                    download_task = progress.add_task(f"Downloading {split} set...", total=None)

                    logger.info(f"Downloading {split} set from {url}")
                    urllib.request.urlretrieve(
                        url,
                        zip_path,
                        reporthook=lambda count, block_size, total_size: progress.update(
                            download_task,
                            total=total_size // block_size if total_size > 0 else None,
                            completed=count,
                        ),
                    )
                    progress.update(download_task, completed=True)

                    # Extract file
                    progress.add_task(f"Extracting {split} set...", total=100)

                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_path)

                    # Clean up
                    zip_path.unlink()
                    logger.info(f"Successfully processed {split} set")
                else:
                    logger.info(f"Found existing {split} set at {extract_path}")

        # Add to cache index
        self.cache_manager.add_to_cache(
            "mind",
            self.version,
            "dataset",
            metadata={
                "splits": list(self.urls.keys()),
                "max_title_length": self.max_title_length,
                "max_history_length": self.max_history_length,
            },
        )

    def process_news(
        self, news_df: pd.DataFrame, split: str = "train_val"
    ) -> Dict[str, np.ndarray]:
        """Process news articles into numerical format"""
        # Check if processed news exists
        preprocessed_path = self.dataset_path / "preprocessed"
        news_cache_path = preprocessed_path / f"processed_news_{split}.pkl"

        if news_cache_path.exists():
            try:
                logger.info(f"Loading processed {split} news from cache...")
                with open(news_cache_path, "rb") as f:
                    processed_news = pickle.load(f)
                logger.info(
                    f"Loaded {len(processed_news['news_ids'])} processed news articles from cache"
                )
                return processed_news
            except (pickle.UnpicklingError, EOFError) as e:
                logger.warning(f"Corrupted news cache detected: {e}")
                news_cache_path.unlink(missing_ok=True)

        # Process news if cache doesn't exist or was corrupted
        processed_news = self._process_news_data(news_df)

        # Save processed news to cache
        logger.info(f"Saving {split} news to cache...")
        with open(news_cache_path, "wb") as f:
            pickle.dump(processed_news, f)
        logger.info(f"Saved {len(processed_news['news_ids'])} news articles to cache")

        return processed_news

    def _process_news_data(self, news_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process news articles into numerical format"""
        if self.embedding_type == "glove":
            # Process with GloVe and store multiple representations
            processed_title_data, title_vocab = self.process_and_tokenize_texts(
                self.max_title_length, news_df, "title"
            )
            processed_abstract_data, abstract_vocab = self.process_and_tokenize_texts(
                self.max_abstract_length, news_df, "abstract"
            )

            processed_news = {
                "news_ids": news_df["id"].values,
                "title": processed_title_data["embeddings"],
                "title_raw": processed_title_data["raw_text"],
                "title_tokens": processed_title_data["token_ids"],
                "abstract": processed_abstract_data["embeddings"],
                "abstract_raw": processed_abstract_data["raw_text"],
                "abstract_tokens": processed_abstract_data["token_ids"],
                "title_vocab": title_vocab,
                "abstract_vocab": abstract_vocab,
            }

            return processed_news
        else:
            # Process with BERT
            title_embeddings = self.embeddings_manager.get_bert_embeddings(
                news_df["title"].tolist(), max_length=self.max_title_length
            )
            abstract_embeddings = self.embeddings_manager.get_bert_embeddings(
                news_df["abstract"].tolist(), max_length=self.max_abstract_length
            )

            return {
                "news_ids": news_df["id"].values,
                "title": title_embeddings,
                "abstract": abstract_embeddings,
            }

    def _get_glove_embeddings(self, texts: pd.Series) -> np.ndarray:
        """Convert texts to GloVe embeddings with better tokenization and detokenization"""
        tokenizer = TreebankWordTokenizer()
        detokenizer = TreebankWordDetokenizer()
        embeddings = []

        for text in texts:
            if pd.isna(text):
                embeddings.append([np.zeros(self.embedding_size)] * self.max_title_length)
                continue

            try:
                # Use TreebankWordTokenizer for better tokenization
                word_tokens = tokenizer.tokenize(text.lower())[: self.max_title_length]

                # Store the detokenized version for potential later use
                detokenized_text = detokenizer.detokenize(word_tokens)

                # Get embeddings for each token
                word_embeddings = [
                    self.embeddings.get(word, np.zeros(self.embedding_size)) for word in word_tokens
                ]

                # Pad if necessary
                if len(word_embeddings) < self.max_title_length:
                    word_embeddings.extend(
                        [np.zeros(self.embedding_size)]
                        * (self.max_title_length - len(word_embeddings))
                    )

                embeddings.append(word_embeddings)

            except Exception as e:
                logger.error(f"Error processing text: {text}")
                logger.error(f"Error details: {str(e)}")
                raise e

        return np.array(embeddings)

    def process_behaviors(
        self, behaviors_df: pd.DataFrame, news_dict: Dict[str, int], stage: str
    ) -> Dict[str, np.ndarray]:
        """Process user behaviors using ImpressionSampler for sampling."""
        logger.info(f"Processing user behaviors for {stage} data...")

        histories = []
        impression_ids = []
        labels = []
        masks = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Processing {stage} behaviors...", total=len(behaviors_df))

            for _, row in behaviors_df.iterrows():
                # Process history
                history = row["history"].split() if pd.notna(row["history"]) else []
                history_indices = [news_dict.get(h, 0) for h in history]
                history_indices = history_indices[-self.max_history_length :]
                history_indices = [0] * (
                    self.max_history_length - len(history_indices)
                ) + history_indices

                # Process impressions using ImpressionSampler
                impressions = row["impressions"].split()
                if self.sampler:
                    impression_groups = self.sampler.sample_impressions(impressions)
                else:
                    # If no sampler, use all impressions as one group
                    impression_groups = [impressions[: self.max_impressions_length]]

                # Process all impression groups
                for group in impression_groups:
                    imp_ids = []
                    click_status = []
                    for imp in group:
                        news_id, label = imp.split("-")
                        imp_ids.append(news_dict.get(news_id, 0))
                        click_status.append(int(label))

                    # Add padding if necessary
                    pad_length = self.max_impressions_length - len(imp_ids)
                    if pad_length > 0:
                        imp_ids.extend([0] * pad_length)
                        click_status.extend([0] * pad_length)

                    histories.append(history_indices)
                    impression_ids.append(imp_ids)
                    labels.append(click_status)
                    masks.append([1] * len(group) + [0] * pad_length)

                progress.advance(task)

        return {
            "histories": np.array(histories),
            "impressions": np.array(impression_ids),
            "labels": np.array(labels),
            "masks": np.array(masks),
        }

    def get_train_val_data(
        self,
    ) -> Tuple[
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    ]:
        """Load and process training data"""
        # Process news data
        news_df = pd.read_csv(
            self.dataset_path / "train" / "news.tsv",
            sep="\t",
            header=None,
            names=[
                "id",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ],
        )
        logger.info(f"News data shape: {news_df.shape}")
        train_val_news_data = self.process_news(news_df, "train_val")  # Store all news features

        # Process behaviors
        behaviors_df = pd.read_csv(
            self.dataset_path / "train" / "behaviors.tsv",
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )
        # Convert time to datetime
        behaviors_df["time"] = pd.to_datetime(behaviors_df["time"])

        # Split into train and validation based on the last day
        last_day = behaviors_df["time"].max().date()
        train_behaviors = behaviors_df[behaviors_df["time"].dt.date < last_day]
        val_behaviors = behaviors_df[behaviors_df["time"].dt.date == last_day]

        logger.info(f"Train behaviors: {len(train_behaviors):,}")
        logger.info(f"Validation behaviors: {len(val_behaviors):,}")

        # Process train behaviors
        train_behaviors_data = self.process_behaviors(
            train_behaviors,
            {nid: idx for idx, nid in enumerate(train_val_news_data["news_ids"])},
            stage="train",
        )

        # Process validation behaviors
        val_behaviors_data = self.process_behaviors(
            val_behaviors,
            {nid: idx for idx, nid in enumerate(train_val_news_data["news_ids"])},
            stage="validation",
        )

        return train_val_news_data, train_behaviors_data, val_behaviors_data

    def get_test_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load and process test data"""
        test_dir = self.dataset_path / "valid"  # Use valid folder for testing

        news_df = pd.read_csv(
            test_dir / "news.tsv",
            sep="\t",
            header=None,
            names=[
                "id",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ],
        )
        test_news_data = self.process_news(news_df, "test")

        behaviors_df = pd.read_csv(
            test_dir / "behaviors.tsv",
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        # Process test behaviors
        test_behaviors_data = self.process_behaviors(
            behaviors_df,
            {nid: idx for idx, nid in enumerate(test_news_data["news_ids"])},
            stage="test",
        )

        return test_news_data, test_behaviors_data

    def train_dataloader(self, batch_size: int) -> Iterator[Tuple]:
        """Create batches of training data."""
        return self._create_dataloader(
            self.train_val_news_data,
            self.train_behaviors_data["histories"],
            self.train_behaviors_data["impressions"],
            self.train_behaviors_data["labels"],
            self.train_behaviors_data["masks"],
            batch_size,
            shuffle=True,
        )

    def val_dataloader(self, batch_size: int) -> Iterator[Tuple]:
        """Create batches of validation data."""
        return self._create_dataloader(
            self.train_val_news_data,
            self.val_behaviors_data["histories"],
            self.val_behaviors_data["impressions"],
            self.val_behaviors_data["labels"],
            self.val_behaviors_data["masks"],
            batch_size,
            shuffle=False,
        )

    def test_dataloader(self, batch_size: int) -> Iterator[Tuple]:
        """Create batches of test data."""
        return self._create_dataloader(
            self.test_news_data,
            self.test_behaviors_data["histories"],
            self.test_behaviors_data["impressions"],
            self.test_behaviors_data["labels"],
            self.test_behaviors_data["masks"],
            batch_size,
            shuffle=False,
        )

    def _create_dataloader(
        self,
        news_data: Dict[str, np.ndarray],
        histories: np.ndarray,  # [n_behaviors, max_history_len]
        impressions: np.ndarray,  # [n_behaviors, max_impressions_len]
        labels: np.ndarray,  # [n_behaviors, max_impressions_len]
        masks: np.ndarray,  # [n_behaviors, max_impressions_len]
        batch_size: int,
        shuffle: bool = False,
    ) -> Iterator[Tuple]:
        """Create batches for training/validation.

        Args:
            histories: User history news IDs
            impressions: Candidate news IDs
            labels: Click labels
            masks: Valid position masks
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Yields:
            Tuple of ((impression_news, history_news), labels, masks)
        """
        n_samples = len(histories)
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Get batch data
            history_batch = histories[batch_indices]  # [batch_size, history_len]
            impressions_batch = impressions[batch_indices]  # [batch_size, impression_len]
            impression_labels = labels[batch_indices]  # [batch_size, impression_len]
            impression_masks = masks[batch_indices]  # [batch_size, impression_len]

            # Get embeddings for history news
            history_news = {
                "title": np.stack(
                    [
                        [
                            (
                                news_data["title"][news_id]
                                if news_id != 0
                                else np.zeros_like(news_data["title"][0])
                            )  # Zero embedding for padding
                            for news_id in user_history
                        ]
                        for user_history in history_batch
                    ]
                )  # [batch_size, history_len, seq_len, emb_dim]
            }

            # Get embeddings for impression news
            impression_news = {
                "title": np.stack(
                    [
                        [
                            (
                                news_data["title"][news_id]
                                if news_id != 0
                                else np.zeros_like(news_data["title"][0])
                            )  # Zero embedding for padding
                            for news_id in user_impressions
                        ]
                        for user_impressions in impressions_batch
                    ]
                )  # [batch_size, impression_len, seq_len, emb_dim]
            }

            yield (impression_news, history_news), impression_labels, impression_masks

    def _apply_data_fraction(self, mode: str = "train") -> None:
        """Reduce the dataset size based on the data_fraction parameter."""
        logger.info(f"Using {self.data_fraction * 100:.0f}% of the dataset for testing purposes.")

        def reduce_data(data_dict: Dict[str, np.ndarray]) -> None:
            """Select data"""
            for key in data_dict:
                data_dict[key] = data_dict[key][: int(len(data_dict[key]) * self.data_fraction)]
    
        if mode == "train":
            reduce_data(self.train_behaviors_data)
            reduce_data(self.val_behaviors_data)
        else:
            reduce_data(self.test_behaviors_data)

    def _display_statistics(self, mode: str = "train") -> None:
        """Display statistics about the dataset."""
        logger.info("Displaying dataset statistics...")

        if mode == "train":
            num_train_val_news = len(self.train_val_news_data["news_ids"])
            num_train_behaviors = len(self.train_behaviors_data["histories"])
            num_val_behaviors = len(self.val_behaviors_data["histories"])

            logger.info(f"Number of news articles: {num_train_val_news}")
            logger.info(f"Number of training behaviors: {num_train_behaviors}")
            logger.info(f"Number of validation behaviors: {num_val_behaviors}")

            # Additional statistics
            avg_history_length = np.mean([len(h) for h in self.train_behaviors_data["histories"]])
            avg_impressions_length = np.mean(
                [len(i) for i in self.train_behaviors_data["impressions"]]
            )
            avg_history_length_val = np.mean([len(h) for h in self.val_behaviors_data["histories"]])
            avg_impressions_length_val = np.mean(
                [len(i) for i in self.val_behaviors_data["impressions"]]
            )

            logger.info(f"Average history length: {avg_history_length:.2f}")
            logger.info(f"Average impressions length: {avg_impressions_length:.2f}")
            logger.info(f"Average history length (validation): {avg_history_length_val:.2f}")
            logger.info(
                f"Average impressions length (validation): {avg_impressions_length_val:.2f}"
            )
        else:
            num_test_news = len(self.test_news_data["news_ids"])
            num_test_behaviors = len(self.test_behaviors_data["histories"])

            logger.info(f"Number of news articles: {num_test_news}")
            logger.info(f"Number of test behaviors: {num_test_behaviors}")

            # Additional statistics
            avg_history_length_test = np.mean(
                [len(h) for h in self.test_behaviors_data["histories"]]
            )
            avg_impressions_length_test = np.mean(
                [len(i) for i in self.test_behaviors_data["impressions"]]
            )

            logger.info(f"Average history length (test): {avg_history_length_test:.2f}")
            logger.info(f"Average impressions length (test): {avg_impressions_length_test:.2f}")

    def process_and_tokenize_texts(
        self, title_len: int, news_df: pd.DataFrame, column_name: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """Process and tokenize texts while building vocabulary and storing both raw and embedded forms."""
        vocab = {"[PAD]": 0, "[UNK]": 1}
        tokenizer = TreebankWordTokenizer()
        detokenizer = TreebankWordDetokenizer()

        processed_data = {
            "news_ids": news_df["id"].values,
            "raw_text": [],
            "token_ids": [],
            "embeddings": [],
        }

        # Use logger instead of print
        logger.info(f"Processing {column_name} for {len(news_df)} articles")

        # Process without Progress bar
        for title in news_df[column_name]:
            if pd.isna(title):
                processed_data["raw_text"].append("")
                processed_data["token_ids"].append([0] * title_len)
                processed_data["embeddings"].append([np.zeros(self.embedding_size)] * title_len)
                continue

            # Tokenize and truncate
            word_tokens = tokenizer.tokenize(title.lower())[:title_len]

            # Store detokenized form
            detokenized_text = detokenizer.detokenize(word_tokens)
            processed_data["raw_text"].append(detokenized_text)

            # Process tokens and build vocabulary
            token_ids = []
            embeddings = []

            for word in word_tokens:
                if word not in vocab:
                    vocab[word] = len(vocab)
                token_ids.append(vocab[word])

                # Get GloVe embedding
                embedding = self.embeddings.get(word, np.zeros(self.embedding_size))
                embeddings.append(embedding)

            # Pad sequences
            pad_length = title_len - len(word_tokens)
            if pad_length > 0:
                token_ids.extend([0] * pad_length)
                embeddings.extend([np.zeros(self.embedding_size)] * pad_length)

            processed_data["token_ids"].append(token_ids)
            processed_data["embeddings"].append(embeddings)

        # Convert to numpy arrays
        processed_data["token_ids"] = np.array(processed_data["token_ids"])
        processed_data["embeddings"] = np.array(processed_data["embeddings"])

        logger.info(f"Processed {len(processed_data['embeddings'])} {column_name} entries")
        logger.info(f"Vocabulary size: {len(vocab)}")

        return processed_data, vocab

import logging
import pickle
import urllib.request
import zipfile
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
import tensorflow as tf

from datasets.base import BaseNewsDataset
from utils.cache_manager import CacheManager
from utils.embeddings import EmbeddingsManager
from utils.logging import setup_logging
from utils.sampling import ImpressionSampler
from datasets.utils import display_statistics, apply_data_fraction
from datasets.dataloader import NewsDataLoader

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
        data_fraction_train: float = 1.0,
        data_fraction_val: float = 1.0,
        data_fraction_test: float = 1.0,
        mode: str = "train",
    ):
        super().__init__()
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
        self.sampler = ImpressionSampler(sampling)
        self.data_fraction_train = data_fraction_train
        self.data_fraction_val = data_fraction_val
        self.data_fraction_test = data_fraction_test
        self.mode = mode

        logger.info(f"Initializing MIND dataset ({version} version)")
        logger.info(f"Data will be stored in: {self.dataset_path}")

        # Create directories if they don't exist
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # Datasets - TensorFlow tensors
        self.train_val_news_data: Dict[str, tf.Tensor] = {}
        self.train_behaviors_data: Dict[str, tf.Tensor] = {}
        self.val_behaviors_data: Dict[str, tf.Tensor] = {}
        self.test_news_data: Dict[str, tf.Tensor] = {}
        self.test_behaviors_data: Dict[str, tf.Tensor] = {}

        # Initialize tokenizer
        self.tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()

        # Set random seed
        np.random.seed(random_seed)

        # Load and process news data (will build vocab and tokenize)
        self.processed_news = self.process_news()

        # Load and process behaviors data
        self._load_data(mode)

    @property
    def train_size(self) -> int:
        """Number of training behaviors"""
        return (
            len(self.train_behaviors_data["labels"]) if hasattr(self, "train_behaviors_data") else 0
        )

    @property
    def val_size(self) -> int:
        """Number of validation behaviors"""
        return len(self.val_behaviors_data["labels"]) if hasattr(self, "val_behaviors_data") else 0

    @property
    def test_size(self) -> int:
        """Number of test behaviors"""
        return (
            len(self.test_behaviors_data["labels"]) if hasattr(self, "test_behaviors_data") else 0
        )

    def process_news(self) -> Dict[str, np.ndarray]:
        """Process news articles into numerical format"""
        processed_path = self.dataset_path / "processed"
        os.makedirs(processed_path, exist_ok=True)

        tokens_file = processed_path / "processed_news.pkl"
        embeddings_file = processed_path / "filtered_embeddings.npy"

        # Check if data has alread been downloaded
        self.download_dataset()

        if not tokens_file.exists() or not embeddings_file.exists():
            logger.info("Processing news data...")
            news_df = self.get_news()

            # Build vocabulary from news titles
            logger.info("Building vocabulary from news titles...")
            vocab = {"[PAD]": 0}
            token_id = 1

            for text in news_df["title"].values:
                words = self.tokenizer.tokenize(text.lower())
                for word in words:
                    if word not in vocab:
                        vocab[word] = token_id
                        token_id += 1

            self.vocab = vocab
            logger.info(f"Vocabulary size: {len(vocab)}")

            # Tokenize all titles
            logger.info("Tokenizing news titles...")
            tokenized_titles = []
            for text in news_df["title"].values:
                tokens = self.tokenize_text(text)
                tokenized_titles.append(tokens)

            # Create filtered embedding matrix
            logger.info("Creating filtered embedding matrix...")
            self.embeddings_manager.load_glove(self.embedding_size)
            filtered_embeddings = self.embeddings_manager.create_filtered_embedding_matrix(
                self.vocab, self.embedding_size
            )

            # Save embeddings
            np.save(embeddings_file, filtered_embeddings.numpy())

            processed_news = {
                "news_ids": news_df["id"].values,
                "tokens": np.array(tokenized_titles, dtype=np.int32),
                "vocab": self.vocab,
                "vocab_size": len(self.vocab),
            }

            with open(tokens_file, "wb") as f:
                pickle.dump(processed_news, f)

        else:
            logger.info("Loading processed news data...")
            with open(tokens_file, "rb") as f:
                processed_news = pickle.load(f)
                self.vocab = processed_news.get("vocab", {"[PAD]": 0})

        # Load filtered embeddings
        if embeddings_file.exists():
            logger.info("Loading filtered embeddings...")
            filtered_embeddings = np.load(embeddings_file)
            processed_news["embeddings"] = tf.constant(filtered_embeddings, dtype=tf.float32)

        return processed_news

    def _load_data(self, mode: str = "train") -> bool:
        """Try to load processed tensor data from disk.

        Returns:
            bool: True if tensors were loaded successfully, False otherwise
        """
        # ------ Data preprocessing and tensor generation ------
        processed_path = self.dataset_path / "processed"
        files_exist = (
            (processed_path / "processed_train.pkl").exists()
            and (processed_path / "processed_val.pkl").exists()
            and (processed_path / "processed_test.pkl").exists()
        )

        if not files_exist:
            self._process_data()

        # ------ Data retrieval ------
        logger.info("Files have already been processed, loading data...")
        try:
            if mode == "train":
                logger.info("Loading train behaviors data...")
                self.train_behaviors_data = pd.read_pickle(processed_path / "processed_train.pkl")
                logger.info("Loading validation behaviors data...")
                self.val_behaviors_data = pd.read_pickle(processed_path / "processed_val.pkl")

                # Apply data fraction if needed
                if self.data_fraction_train < 1.0:
                    self.train_behaviors_data = apply_data_fraction(
                        self.train_behaviors_data, self.data_fraction_train
                    )
                
                if self.data_fraction_val < 1.0:
                    self.val_behaviors_data = apply_data_fraction(
                        self.val_behaviors_data, self.data_fraction_val
                    )

                # Display statistics
                self._display_statistics(
                    mode,
                    processed_news=self.processed_news,
                    train_behaviors_data=self.train_behaviors_data,
                    val_behaviors_data=self.val_behaviors_data,
                )

                logger.info("Successfully loaded processed train/val tensors")
                return True

            else:  # test mode
                logger.info("Loading test behaviors data...")
                self.test_behaviors_data = pd.read_pickle(processed_path / "processed_test.pkl")

                # Apply data fraction if needed
                if self.data_fraction_test < 1.0:
                    self.test_behaviors_data = apply_data_fraction(
                        self.test_behaviors_data, self.data_fraction_test
                    )

                # Display statistics
                self._display_statistics(
                    mode,
                    processed_news=self.processed_news,
                    test_behaviors_data=self.test_behaviors_data,
                )

                logger.info("Successfully loaded processed test tensors")
                return True

        except Exception as e:
            logger.warning(f"Failed to load tensors: {str(e)}")
            return False

    def _process_data(self) -> None:
        """Process train/val/test data and save to disk."""
        processed_path = self.dataset_path / "processed"

        # ----- Process train data
        logger.info("Processing train data...")
        train_behaviors_dict, val_behaviors_dict = self.get_train_val_data()
        # Save train data
        logger.info("Saving training data...")
        with open(processed_path / "processed_train.pkl", "wb") as f:
            pickle.dump(train_behaviors_dict, f)

        # ----- Process validation data
        logger.info("Processing validation data...")
        # Save val data
        with open(processed_path / "processed_val.pkl", "wb") as f:
            pickle.dump(val_behaviors_dict, f)

        # ----- Process testing data
        logger.info("Processing test data...")
        test_behaviors_dict = self.get_test_data()

        # Save test data
        logger.info("Saving test data...")
        with open(processed_path / "processed_test.pkl", "wb") as f:
            pickle.dump(test_behaviors_dict, f)

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

    def process_behaviors(self, behaviors_df: pd.DataFrame, stage: str) -> Dict[str, np.ndarray]:
        """Process behaviors storing only tokens, not embeddings."""
        histories = []
        history_tokens = []
        histories_masks = []
        impression_ids = []
        impression_tokens = []
        labels = []
        impression_masks = []  # New mask for impressions

        # Statistics tracking
        total_original_rows = len(behaviors_df)
        total_positives = 0
        rows_with_multiple_positives = 0
        max_positives_in_row = 0
        max_impressions_length = 0  # Track maximum impressions length

        # Create a lookup dictionary from news ID to pre-tokenized titles
        news_tokens = dict(
            zip(
                [int(nid.split("N")[1]) for nid in self.processed_news["news_ids"]],
                self.processed_news["tokens"],
            )
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Processing {stage} behaviors...", total=len(behaviors_df))

            # First pass: find maximum impressions length
            for _, row in behaviors_df.iterrows():
                impressions = row["impressions"].split()
                max_impressions_length = max(max_impressions_length, len(impressions))

            # Reset progress for second pass
            progress.reset(task)

            # Parse all behavior rows
            for _, row in behaviors_df.iterrows():
                # Count positives in this row
                impressions = row["impressions"].split()
                positives_count = sum(1 for imp in impressions if imp.split("-")[1] == "1")

                total_positives += positives_count
                if positives_count > 1:
                    rows_with_multiple_positives += 1
                max_positives_in_row = max(max_positives_in_row, positives_count)

                # -- Process history
                history = row["history"].split() if pd.notna(row["history"]) else []
                history = history[-self.max_history_length :]

                # -- Store indices/tokens instead of embeddings
                history_indices = [int(h.split("N")[1]) for h in history]
                curr_history_tokens = [news_tokens[h_idx] for h_idx in history_indices]

                # -- Pad histories
                history_pad_length = self.max_history_length - len(history)
                history_indices = [0] * history_pad_length + history_indices
                curr_history_tokens = [
                    [0] * self.max_title_length
                ] * history_pad_length + curr_history_tokens
                history_mask = [0] * history_pad_length + [1] * len(history)

                # -- Process impressions
                # imp_groups: List of impression groups, where each group contains:
                #   - 1 positive news article
                #   - k negative news articles (k = max_impressions_length - 1)
                #   - Articles are shuffled within each group
                # label_groups: List of label groups corresponding to imp_groups, where:
                #   - 1 represents a positive article
                #   - 0 represents a negative article
                imp_groups, label_groups = self.sampler.sample_impressions(
                    impressions,
                    is_training=(stage == "train")
                )

                if stage == "train":
                    # -- For each group (1 positive : k negatives)
                    # Some rows have multiple positive articles, so we need to repeat the same history for each group
                    for imp_group, label_group in zip(imp_groups, label_groups):
                        # Repeat the same history for this group
                        histories.append(history_indices)
                        history_tokens.append(curr_history_tokens)
                        histories_masks.append(history_mask)

                        # Get tokens for impressions using pre-tokenized version
                        curr_impression_tokens = [news_tokens[nid] for nid in imp_group]

                        impression_ids.append(imp_group)
                        impression_tokens.append(curr_impression_tokens)
                        labels.append(label_group)
                        impression_masks.append([1] * len(imp_group))  # All positions are valid
                else:
                    # -- For validation/testing, handle variable length impressions
                    # Just change the word to singular to convery that we're dealing with a single group with all impressions
                    imp_group, label_group = imp_groups, label_groups 
                    
                    # Pad impressions to max length
                    pad_length = max_impressions_length - len(imp_group)
                    padded_impressions = imp_group + [0] * pad_length
                    padded_labels = label_group + [0] * pad_length
                    padded_tokens = [news_tokens[nid] for nid in imp_group] + [
                        [0] * self.max_title_length
                    ] * pad_length
                    
                    # Create mask for valid positions
                    impression_mask = [1] * len(imp_group) + [0] * pad_length

                    histories.append(history_indices)
                    history_tokens.append(curr_history_tokens)
                    histories_masks.append(history_mask)
                    impression_ids.append(padded_impressions)
                    impression_tokens.append(padded_tokens)
                    labels.append(padded_labels)
                    impression_masks.append(impression_mask)

                progress.advance(task)

        # Convert to numpy arrays
        result = {
            "histories": np.array(histories, dtype=np.int32),
            "history_tokens": np.array(history_tokens, dtype=np.int32),
            "histories_masks": np.array(histories_masks, dtype=np.float32),
            "impressions": np.array(impression_ids, dtype=np.int32),
            "impression_tokens": np.array(impression_tokens, dtype=np.int32),
            "labels": np.array(labels, dtype=np.float32),
            "impression_masks": np.array(impression_masks, dtype=np.float32),
        }

        # -- Calculate statistics
        total_processed_rows = len(histories)
        expansion_factor = total_processed_rows / total_original_rows

        # -- Log statistics
        logger.info(f"\nBehavior Processing Statistics ({stage}):")
        logger.info(f"Original number of rows: {total_original_rows:,}")
        logger.info(f"Processed number of rows: {total_processed_rows:,}")
        logger.info(f"Expansion factor: {expansion_factor:.2f}x")
        logger.info(f"Total positive samples: {total_positives:,}")
        logger.info(
            f"Rows with multiple positives: {rows_with_multiple_positives:,} "
            f"({rows_with_multiple_positives/total_original_rows*100:.1f}%)"
        )
        logger.info(f"Maximum positives in a single row: {max_positives_in_row}")
        logger.info(f"Average positives per row: {total_positives/total_original_rows:.2f}")
        logger.info(f"Maximum impressions length: {max_impressions_length}")

        return result

    def get_news(self) -> Tuple[Dict[str, np.ndarray]]:
        """Load and process news data"""
        train_news_df = pd.read_csv(
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

        val_news_df = pd.read_csv(
            self.dataset_path / "valid" / "news.tsv",
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

        news_df = pd.concat([train_news_df, val_news_df])
        news_df["news_ids"] = news_df["id"].values
        logger.info(f"News data shape: {news_df.shape}")

        return news_df

    def get_train_val_data(
        self,
    ) -> Tuple[
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    ]:
        """Load and process training data"""

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

        # Process train/valid behaviors
        train_behaviors_data = self.process_behaviors(
            train_behaviors,
            stage="train",
        )

        logger.info(f"Validation behaviors: {len(val_behaviors):,}")

        val_behaviors_data = self.process_behaviors(
            val_behaviors,
            stage="val",
        )

        return train_behaviors_data, val_behaviors_data

    def get_test_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load and process test data"""
        test_dir = self.dataset_path / "valid"  # Use valid folder for testing

        test_behaviors = pd.read_csv(
            test_dir / "behaviors.tsv",
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        logger.info(f"Test behaviors: {len(test_behaviors):,}")

        # Process test behaviors
        test_behaviors_data = self.process_behaviors(
            test_behaviors,
            stage="test",
        )

        return test_behaviors_data

    def train_dataloader(self, batch_size: int) -> tf.data.Dataset:
        """Create training dataset with token-based inputs."""
        return NewsDataLoader.create_train_dataset(
            history_tokens=tf.convert_to_tensor(self.train_behaviors_data["history_tokens"], dtype=tf.int32),
            impression_tokens=tf.convert_to_tensor(
                self.train_behaviors_data["impression_tokens"], dtype=tf.int32
            ),
            labels=tf.convert_to_tensor(self.train_behaviors_data["labels"], dtype=tf.float32),
            histories_masks=tf.convert_to_tensor(
                self.train_behaviors_data["histories_masks"], dtype=tf.float32
            ),
            impression_masks=tf.convert_to_tensor(
                self.train_behaviors_data["impression_masks"], dtype=tf.float32
            ),
            batch_size=batch_size,
        )

    def val_dataloader(self, batch_size: int) -> tf.data.Dataset:
        """Create validation dataset with token-based inputs."""
        return NewsDataLoader.create_eval_dataset(
            history_tokens=tf.convert_to_tensor(
                self.val_behaviors_data["history_tokens"], dtype=tf.int32
            ),
            impression_tokens=tf.convert_to_tensor(
                self.val_behaviors_data["impression_tokens"], dtype=tf.int32
            ),
            labels=tf.convert_to_tensor(self.val_behaviors_data["labels"], dtype=tf.float32),
            histories_masks=tf.convert_to_tensor(
                self.val_behaviors_data["histories_masks"], dtype=tf.float32
            ),
            impression_masks=tf.convert_to_tensor(
                self.val_behaviors_data["impression_masks"], dtype=tf.float32
            ),
            batch_size=batch_size,
        )

    def test_dataloader(self, batch_size: int) -> tf.data.Dataset:
        """Create test dataset with token-based inputs."""
        return NewsDataLoader.create_eval_dataset(
            history_tokens=tf.convert_to_tensor(
                self.test_behaviors_data["history_tokens"], dtype=tf.int32
            ),
            impression_tokens=tf.convert_to_tensor(
                self.test_behaviors_data["impression_tokens"], dtype=tf.int32
            ),
            labels=tf.convert_to_tensor(self.test_behaviors_data["labels"], dtype=tf.float32),
            histories_masks=tf.convert_to_tensor(
                self.test_behaviors_data["histories_masks"], dtype=tf.float32
            ),
            impression_masks=tf.convert_to_tensor(
                self.test_behaviors_data["impression_masks"], dtype=tf.float32
            ),
            batch_size=batch_size,
        )

    def _display_statistics(
        self,
        mode: str = "train",
        processed_news: Dict[str, np.ndarray] = None,
        train_behaviors_data: Dict[str, np.ndarray] = None,
        val_behaviors_data: Dict[str, np.ndarray] = None,
        test_behaviors_data: Dict[str, np.ndarray] = None,
    ) -> None:
        if mode == "train":
            data_dict = {
                "news": processed_news,
                "train_behaviors": train_behaviors_data,
                "val_behaviors": val_behaviors_data,
            }
        else:
            data_dict = {
                "news": processed_news,
                "test_behaviors": test_behaviors_data,
            }
        display_statistics(data_dict, mode)

    def tokenize_text(self, text: str) -> List[int]:
        """Convert text to a list of token IDs using NLTK's TreebankWordTokenizer."""
        # Tokenize text into words using NLTK
        words = self.tokenizer.tokenize(text.lower())

        # Convert words to token IDs, using 0 ([PAD]) for unknown words
        tokens = [self.vocab.get(word, 0) for word in words]

        # Pad or truncate to max_title_length
        tokens = tokens[: self.max_title_length]
        tokens.extend([0] * (self.max_title_length - len(tokens)))

        return tokens

    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Build vocabulary from texts."""
        vocab = {"[PAD]": 0}
        token_id = 1

        for text in texts:
            words = self.tokenizer.tokenize(text.lower())
            for word in words:
                if word not in vocab:
                    vocab[word] = token_id
                    token_id += 1

        return vocab

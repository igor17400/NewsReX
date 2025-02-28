import logging
import pickle
import urllib.request
import zipfile
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from tensorflow.keras.preprocessing.text import Tokenizer

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
        root_dir: str,
        urls: Dict,
        max_title_length: int,
        max_abstract_length: int,
        max_history_length: int,
        max_impressions_length: int,
        min_word_freq: int,
        max_vocabulary_size: int,
        random_seed: int,
        embedding_type: str = "glove",
        embedding_size: int = 300,
        sampling: DictConfig = None,
        data_fraction: float = 0.005,
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
        self.min_word_freq = min_word_freq
        self.max_vocabulary_size = max_vocabulary_size
        self.embedding_type = embedding_type
        self.embedding_size = embedding_size
        self.embeddings_manager = EmbeddingsManager(self.cache_manager)
        self.sampler = ImpressionSampler(sampling) if sampling else None
        self.data_fraction = data_fraction

        logger.info(f"Initializing MIND dataset ({version} version)")
        logger.info(f"Data will be stored in: {self.dataset_path}")

        # Create directories if they don't exist
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # Set random seed
        np.random.seed(random_seed)

        # Load and process data
        self._load_data()

    def _load_data(self) -> None:
        """Load and process all data."""
        # Check if processed data exists
        processed_data_path = self.dataset_path / "processed_data.pkl"
        if processed_data_path.exists():
            logger.info("Loading processed data from disk...")
            with open(processed_data_path, "rb") as f:
                self.news_data, self.behaviors_data, self.val_behaviors_data = pickle.load(f)
            logger.info("Processed data loaded successfully.")

            # Apply data fraction to reduce dataset size for testing
            if self.data_fraction < 1.0:
                self._apply_data_fraction()

            return

        # Download dataset if needed
        self.download_dataset()

        # Build vocabulary from training data
        self.tokenizer = self.build_vocabulary()

        # Load embeddings
        if self.embedding_type == "glove":
            self.embeddings = self.embeddings_manager.load_glove()
        elif self.embedding_type == "bert":
            self.bert_model, self.bert_tokenizer = self.embeddings_manager.load_bert()

        # Load train and validation data
        (self.news_data, self.behaviors_data), (_, self.val_behaviors_data) = self.get_train_val_data()

        # Apply data fraction to reduce dataset size for testing
        if self.data_fraction < 1.0:
            self._apply_data_fraction()

        # Save processed data
        logger.info("Saving processed data to disk...")
        with open(processed_data_path, "wb") as f:
            pickle.dump((self.news_data, self.behaviors_data, self.val_behaviors_data), f)
        logger.info("Processed data saved successfully.")

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

    def build_vocabulary(self) -> Tokenizer:
        """Build vocabulary from training data"""
        logger.info("Building vocabulary from training data...")

        # Create tokenizer
        tokenizer = Tokenizer(
            num_words=self.max_vocabulary_size,
            oov_token="<UNK>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        )

        # Read news data from both "train" and "valid" directories
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

        valid_news_df = pd.read_csv(
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

        # Concatenate the news data from both directories
        news_df = pd.concat([train_news_df, valid_news_df], ignore_index=True)

        # Combine title and abstract text
        texts = []
        with Progress() as progress:
            task = progress.add_task("Processing texts...", total=len(news_df))
            for _, row in news_df.iterrows():
                title = row["title"] if pd.notna(row["title"]) else ""
                abstract = row["abstract"] if pd.notna(row["abstract"]) else ""
                texts.append(f"{title} {abstract}")
                progress.advance(task)

        # Fit tokenizer
        tokenizer.fit_on_texts(texts)

        return tokenizer

    def process_news(self, news_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process news articles into numerical format"""
        with console.status("[bold green]Processing news articles..."):
            if self.embedding_type == "glove":
                # Process with GloVe
                title_embeddings = self._get_glove_embeddings(news_df["title"])
                abstract_embeddings = self._get_glove_embeddings(news_df["abstract"])
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
        """Convert texts to GloVe embeddings"""
        embeddings = []
        for text in texts:
            if pd.isna(text):
                # Handle NaN by appending a zero vector of the appropriate length
                embeddings.append([np.zeros(self.embedding_size)] * self.max_title_length)
                continue

            try:
                words = text.lower().split()
                word_embeddings = [
                    self.embeddings.get(word, np.zeros(self.embedding_size)) for word in words[: self.max_title_length]
                ]
                # Pad if necessary
                if len(word_embeddings) < self.max_title_length:
                    word_embeddings.extend([np.zeros(self.embedding_size)] * (self.max_title_length - len(word_embeddings)))
                embeddings.append(word_embeddings)
            except Exception as e:
                raise Exception(f"Error processing text: {text} | Error: {e}")
        return np.array(embeddings)

    def process_behaviors(
        self, behaviors_df: pd.DataFrame, news_dict: Dict[str, int], stage: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process user behaviors into numerical format suitable for model input.

        This method processes a DataFrame containing user behavior data, specifically
        the history of news articles a user has interacted with and the impressions
        (news articles shown to the user along with their click status). The method
        converts these into fixed-size numerical arrays that can be used as input
        for machine learning models.

        Parameters:
        ----------
        behaviors_df : pd.DataFrame
            A DataFrame containing user behavior data. It must include the columns:
            - 'history': A string of space-separated news IDs representing the user's
              interaction history.
            - 'impressions': A string of space-separated pairs in the format 'newsID-clickStatus',
              where 'clickStatus' is '1' if the article was clicked and '0' otherwise.

        news_dict : Dict[str, int]
            A dictionary mapping news IDs to numerical indices. This is used to convert
            news IDs into indices that can be used as input for models.

        stage : str
            A string indicating whether the method is being called for training or validation data.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing four numpy arrays:
            - histories: A 2D numpy array where each row represents a user's interaction
              history, converted into indices and padded to a fixed length.
            - impression_ids: A 2D numpy array where each row contains the indices of
              news articles in the impression list, padded to a fixed length.
            - labels: A 2D numpy array where each row contains the click status for each
              impression shown to the user, represented as binary values (1 for clicked,
              0 for not clicked), padded to a fixed length.
            - masks: A 2D numpy array where each row contains binary values indicating
              which positions in the impression list are valid (1) and which are padded (0).
        """
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
                history_indices = [0] * (self.max_history_length - len(history_indices)) + history_indices

                # Process impressions
                impressions = row["impressions"].split()
                if self.sampler:
                    impression_groups = self.sampler.sample_impressions(impressions)
                    # Process all impression groups at once
                    imp_ids_all = []
                    click_status_all = []
                    masks_all = []

                    for group in impression_groups:
                        imp_ids = []
                        click_status = []
                        for imp in group:
                            news_id, label = imp.split("-")
                            imp_ids.append(news_dict.get(news_id, 0))
                            click_status.append(int(label))

                        imp_ids_all.append(imp_ids)
                        click_status_all.append(click_status)
                        masks_all.append([1] * len(group))

                    # Add the history once for all groups
                    histories.extend([history_indices] * len(impression_groups))
                    impression_ids.extend(imp_ids_all)
                    labels.extend(click_status_all)
                    masks.extend(masks_all)
                else:
                    # Original processing for no sampling
                    imp_ids = []
                    click_status = []
                    for imp in impressions:
                        news_id, label = imp.split("-")
                        imp_ids.append(news_dict.get(news_id, 0))
                        click_status.append(int(label))

                    # Pad if necessary
                    mask = [1] * len(imp_ids)
                    if len(imp_ids) < self.max_impressions_length:
                        imp_ids += [0] * (self.max_impressions_length - len(imp_ids))
                        click_status += [0] * (self.max_impressions_length - len(click_status))
                        mask += [0] * (self.max_impressions_length - len(mask))
                    else:
                        imp_ids = imp_ids[: self.max_impressions_length]
                        click_status = click_status[: self.max_impressions_length]
                        mask = mask[: self.max_impressions_length]

                    histories.append(history_indices)
                    impression_ids.append(imp_ids)
                    labels.append(click_status)
                    masks.append(mask)

                progress.advance(task)

        logger.info(f"Processed {len(behaviors_df):,} user behaviors for {stage} data")
        histories_arr = np.array(histories)
        logger.info(f"Histories shape: {histories_arr.shape}")
        impression_ids_arr = np.array(impression_ids)
        logger.info(f"Impression IDs shape: {impression_ids_arr.shape}")
        labels_arr = np.array(labels)
        logger.info(f"Labels shape: {labels_arr.shape}")
        masks_arr = np.array(masks)
        logger.info(f"Masks shape: {masks_arr.shape}")

        return histories_arr, impression_ids_arr, labels_arr, masks_arr

    def get_train_val_data(
        self,
    ) -> Tuple[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
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
        self.news_data = self.process_news(news_df)  # Store all news features

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
        train_histories, train_impressions, train_labels, train_masks = self.process_behaviors(
            train_behaviors,
            {nid: idx for idx, nid in enumerate(self.news_data["news_ids"])},
            stage="train",
        )

        # Store train behaviors data
        self.behaviors_data = {
            "histories": train_histories,
            "impressions": train_impressions,
            "labels": train_labels,
            "masks": train_masks,
        }

        # Process validation behaviors
        val_histories, val_impressions, val_labels, val_masks = self.process_behaviors(
            val_behaviors,
            {nid: idx for idx, nid in enumerate(self.news_data["news_ids"])},
            stage="validation",
        )

        # Store validation behaviors data
        self.val_behaviors_data = {
            "histories": val_histories,
            "impressions": val_impressions,
            "labels": val_labels,
            "masks": val_masks,
        }

        return (self.news_data, self.behaviors_data), (self.news_data, self.val_behaviors_data)

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
        self.test_news_data = self.process_news(news_df)

        behaviors_df = pd.read_csv(
            test_dir / "behaviors.tsv",
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        test_histories, test_impressions, test_labels, test_masks = self.process_behaviors(
            behaviors_df,
            {nid: idx for idx, nid in enumerate(self.test_news_data["news_ids"])},
            stage="test",
        )

        # Store test behaviors data
        self.test_behaviors_data = {
            "histories": test_histories,
            "impressions": test_impressions,
            "labels": test_labels,
            "masks": test_masks,
        }
        self.n_test_behaviors = len(behaviors_df)

        return self.test_news_data, self.test_behaviors_data

    def train_dataloader(self, batch_size: int) -> Iterator[Tuple]:
        """Create batches of training data."""
        return self._create_dataloader(
            self.behaviors_data["histories"],
            self.behaviors_data["impressions"],
            self.behaviors_data["labels"],
            self.behaviors_data["masks"],
            batch_size,
            shuffle=True,
        )

    def val_dataloader(self, batch_size: int) -> Iterator[Tuple]:
        """Create batches of validation data."""
        return self._create_dataloader(
            self.val_behaviors_data["histories"],
            self.val_behaviors_data["impressions"],
            self.val_behaviors_data["labels"],
            self.val_behaviors_data["masks"],
            batch_size,
            shuffle=False,
        )

    def _create_dataloader(
        self,
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
                            self.news_data["title"][news_id]
                            if news_id != 0
                            else np.zeros_like(self.news_data["title"][0])  # Zero embedding for padding
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
                            self.news_data["title"][news_id]
                            if news_id != 0
                            else np.zeros_like(self.news_data["title"][0])  # Zero embedding for padding
                            for news_id in user_impressions
                        ]
                        for user_impressions in impressions_batch
                    ]
                )  # [batch_size, impression_len, seq_len, emb_dim]
            }

            yield (impression_news, history_news), impression_labels, impression_masks

    def _apply_data_fraction(self) -> None:
        """Reduce the dataset size based on the data_fraction parameter."""
        logger.info(f"Using {self.data_fraction * 100:.0f}% of the dataset for testing purposes.")

        def reduce_data(data_dict: Dict[str, np.ndarray]) -> None:
            """Select data"""
            for key in data_dict:
                data_dict[key] = data_dict[key][: int(len(data_dict[key]) * self.data_fraction)]

        reduce_data(self.behaviors_data)
        reduce_data(self.val_behaviors_data)

    @property
    def n_val_behaviors(self) -> int:
        """Number of validation behaviors"""
        return len(self.val_behaviors_data["labels"]) if hasattr(self, "val_behaviors_data") else 0

    @property
    def n_behaviors(self) -> int:
        """Number of training behaviors"""
        return len(self.behaviors_data["labels"]) if hasattr(self, "behaviors_data") else 0

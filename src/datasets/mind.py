import logging
import urllib.request
import zipfile
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from tensorflow.keras.preprocessing.text import Tokenizer

from utils.cache_manager import CacheManager
from utils.embeddings import EmbeddingsManager

# Set up rich logging
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("mind_dataset")
console = Console()


class MINDDataset:
    def __init__(
        self,
        name: str,
        version: str,
        root_dir: str,
        urls: Dict,
        max_title_length: int,
        max_abstract_length: int,
        max_history_length: int,
        min_word_freq: int,
        max_vocabulary_size: int,
        embedding_type: str = "glove",
    ):
        self.name = name
        self.version = version
        self.cache_manager = CacheManager()
        self.dataset_path = self.cache_manager.get_dataset_path("mind", version)
        self.urls = urls[version]
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length
        self.max_history_length = max_history_length
        self.min_word_freq = min_word_freq
        self.max_vocabulary_size = max_vocabulary_size
        self.embedding_type = embedding_type
        self.embeddings_manager = EmbeddingsManager(self.cache_manager)

        logger.info(f"Initializing MIND dataset ({version} version)")
        logger.info(f"Data will be stored in: {self.dataset_path}")

        # Create directories if they don't exist
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # Download and process data
        self.download_dataset()
        self.tokenizer = self.build_vocabulary()

        if embedding_type == "glove":
            self.embeddings = self.embeddings_manager.load_glove()
        elif embedding_type == "bert":
            self.bert_model, self.bert_tokenizer = self.embeddings_manager.load_bert()

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

        news_file = self.dataset_path / "train" / "news.tsv"
        with console.status as status:
            news_df = pd.read_csv(
                news_file,
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

            status.update("[bold green]Processing texts...")
            texts = news_df["title"].fillna("") + " " + news_df["abstract"].fillna("")

            status.update("[bold green]Creating tokenizer...")
            tokenizer = Tokenizer(num_words=self.max_vocabulary_size, oov_token="<UNK>")

            status.update("[bold green]Fitting tokenizer...")
            tokenizer.fit_on_texts(texts)

        vocab_size = len(tokenizer.word_index)
        logger.info(f"Vocabulary built with {vocab_size:,} words")
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
            words = text.lower().split()
            word_embeddings = [
                self.embeddings.get(word, np.zeros(300)) for word in words[: self.max_title_length]
            ]
            # Pad if necessary
            if len(word_embeddings) < self.max_title_length:
                word_embeddings.extend(
                    [np.zeros(300)] * (self.max_title_length - len(word_embeddings))
                )
            embeddings.append(word_embeddings)
        return np.array(embeddings)

    def process_behaviors(
        self, behaviors_df: pd.DataFrame, news_dict: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process user behaviors into numerical format"""
        logger.info("Processing user behaviors...")
        histories = []
        labels = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing behaviors...", total=len(behaviors_df))

            for _, row in behaviors_df.iterrows():
                history = row["history"].split()
                history_indices = [news_dict.get(h, 0) for h in history]
                history_indices = history_indices[-self.max_history_length :]
                history_indices = [0] * (
                    self.max_history_length - len(history_indices)
                ) + history_indices

                impressions = row["impressions"].split()
                for imp in impressions:
                    news_id, label = imp.split("-")
                    histories.append(history_indices)
                    labels.append(int(label))

                progress.advance(task)

        logger.info(f"Processed {len(behaviors_df):,} user behaviors")
        return np.array(histories), np.array(labels)

    def get_train_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Load and process training data"""
        train_dir = self.dataset_path / "train"

        # Load news data
        news_df = pd.read_csv(
            train_dir / "news.tsv",
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
        news_data = self.process_news(news_df)

        # Create news ID mapping
        news_dict = {nid: idx for idx, nid in enumerate(news_data["news_ids"])}

        # Load behaviors
        behaviors_df = pd.read_csv(
            train_dir / "behaviors.tsv",
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        histories, labels = self.process_behaviors(behaviors_df, news_dict)

        return news_data, histories, labels

    def get_validation_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Load and process validation data"""
        valid_dir = self.dataset_path / "valid"

        # Similar to get_train_data()
        news_df = pd.read_csv(
            valid_dir / "news.tsv",
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
        news_data = self.process_news(news_df)

        news_dict = {nid: idx for idx, nid in enumerate(news_data["news_ids"])}

        behaviors_df = pd.read_csv(
            valid_dir / "behaviors.tsv",
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        histories, labels = self.process_behaviors(behaviors_df, news_dict)

        return news_data, histories, labels

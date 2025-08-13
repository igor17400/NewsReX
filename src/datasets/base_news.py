import logging
import pickle
import urllib.request
import zipfile
import os
from typing import Dict, Tuple, List, Optional, Set, Any, Union
import collections
import re
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import shutil

from numpy.f2py.auxfuncs import throw_error
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from src.datasets.base import BaseNewsDataset
from src.utils.data_aux.cache_manager import CacheManager
from src.utils.data_aux.embeddings import EmbeddingsManager
from src.utils.data_aux.bpemb_manager import BPEmbManager
from src.utils.io.logging import setup_logging
from src.utils.data_aux.sampling import ImpressionSampler
from src.datasets.utils import (
    display_statistics,
    apply_data_fraction,
    string_is_number,
    collect_basic_dataset_info,
    collect_news_statistics,
    collect_behavior_statistics,
    collect_overall_statistics,
    collect_quality_metrics,
    reorder_summary_columns,
    log_key_statistics,
    save_unique_users_to_csv,
)
from src.datasets.dataloader import (
    NewsDataLoader,
    UserHistoryBatchDataloader,
    NewsBatchDataloader,
    ImpressionIterator,
)
from src.datasets.knowledge_graph import KnowledgeGraphProcessor
from .custom_dataset_utils import preprocess_custom_dataset

setup_logging()
logger = logging.getLogger(__name__)
console = Console()


def _has_atleast_one_pos_neg_samples(
        impressions_item,
        available_news_ids,
        parse_news_id,
        progress_callback=None
) -> bool:
    """Check if impressions list has at least one positive (label=1) and one negative (label=0) sample."""
    try:
        # Split space-separated impressions
        impressions_list = str(impressions_item).split()

        if len(impressions_list) == 0:
            if progress_callback:
                progress_callback()
            return False

        has_positive = False
        has_negative = False

        # Check for both positive and negative samples
        for item in impressions_list:
            if isinstance(item, str) and '-' in item:
                # Format: "news_id-label"
                parts = item.split('-')
                news_id, label = parts[0], parts[1]
                news_id = parse_news_id(news_id)
                if news_id in available_news_ids:
                    if len(parts) >= 2:
                        if label == '1':
                            has_positive = True
                        elif label == '0':
                            has_negative = True

            # Early exit if we found both
            if has_positive and has_negative:
                if progress_callback:
                    progress_callback()
                return True

        result = has_positive and has_negative
        if progress_callback:
            progress_callback()
        return result

    except Exception as e:
        logger.warning(f"Failed to check positive/negative samples for {impressions_item}: {e}")
        if progress_callback:
            progress_callback()
        return False


class NewsDatasetBase(BaseNewsDataset):
    """Base class for news recommendation datasets following MIND format.
    
    Supports three modes of operation:
    1. Pre-split datasets (with train/valid directories)
    2. Auto-splitting of single behaviors.tsv files (auto_split_behaviors=True)
    3. Auto-conversion of custom formats to MIND format (auto_convert_format=True)
    
    Auto-conversion behavior (auto_convert_format=True):
    - Converts custom dataset formats to MIND format
    - Original files: behaviors.tsv + news.tsv in dataset root
    - Converted files: saved to conversion/ subdirectory
    - Subsequent processing uses converted files
    
    Auto-split behavior (auto_split_behaviors=True):
    - Extracts last day(s) as test data
    - Remaining data is used for train/validation split with existing strategies
    - Test data goes to valid/ directory to maintain compatibility with existing code
    - Train+validation data goes to train/ directory for further splitting
    
    Both auto-conversion and auto-splitting can be used together.
    """

    def __init__(
            self,
            name: str,
            version: str,
            data_path: Optional[str] = None,
            urls: Optional[Dict] = None,
            language: str = "english",
            max_title_length: int = 30,
            max_abstract_length: int = 50,
            max_history_length: int = 50,
            max_impressions_length: int = 5,
            seed: int = 42,
            embedding_type: str = "glove",
            embedding_size: int = 300,
            sampling: Optional[DictConfig] = None,
            data_fraction_train: float = 1.0,
            data_fraction_val: float = 1.0,
            data_fraction_test: float = 1.0,
            mode: str = "train",
            use_knowledge_graph: bool = False,
            random_train_samples: bool = False,
            validation_split_strategy: str = "chronological",
            validation_split_percentage: float = 0.05,
            validation_split_seed: Optional[int] = None,
            auto_split_behaviors: bool = False,  # Auto-split single behaviors.tsv into train/test
            auto_convert_format: bool = False,  # Auto-convert custom format to MIND format
            word_threshold: int = 3,
            process_title: bool = True,
            process_abstract: bool = True,
            process_category: bool = True,
            process_subcategory: bool = True,
            process_user_id: bool = False,
            max_entities: int = 1000,
            max_relations: int = 500,
            download_if_missing: bool = True,
            id_prefix: str = "N",  # Prefix for news IDs (e.g., "N" for MIND)
            user_id_prefix: str = "U",  # Prefix for user IDs (e.g., "U" for MIND)
    ):
        super().__init__()
        self.name = name
        self.version = version
        self.language = language
        self.use_knowledge_graph = use_knowledge_graph
        self.cache_manager = CacheManager()

        # ID prefixes for parsing
        self.id_prefix = id_prefix
        self.user_id_prefix = user_id_prefix

        if data_path:
            self.dataset_path = Path(data_path)
        else:
            self.dataset_path = self.cache_manager.get_dataset_path(name.lower().replace(" ", "_"), version)

        self.urls = urls
        self.download_if_missing = download_if_missing
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length
        self.max_history_length = max_history_length
        self.max_impressions_length = max_impressions_length
        self.embedding_type = embedding_type
        self.embedding_size = embedding_size
        self.embeddings_manager = EmbeddingsManager(self.cache_manager)
        self.bpemb_manager = BPEmbManager(self.cache_manager)
        self.sampler = ImpressionSampler(sampling if sampling is not None else DictConfig({}))
        self.data_fraction_train = data_fraction_train
        self.data_fraction_val = data_fraction_val
        self.data_fraction_test = data_fraction_test
        self.mode = mode
        self.random_train_samples = random_train_samples
        self.word_threshold = word_threshold
        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory
        self.process_user_id = process_user_id

        policy = keras.mixed_precision.global_policy()
        if policy.compute_dtype == "mixed_float16":
            self.float_dtype = "float16"
        elif policy.compute_dtype == "float16":
            self.float_dtype = "float16"
        else:
            self.float_dtype = "float32"

        self.validation_split_strategy = validation_split_strategy
        self.validation_split_percentage = validation_split_percentage
        self.validation_split_seed = (
            validation_split_seed if validation_split_seed is not None else seed
        )
        self.auto_split_behaviors = auto_split_behaviors
        self.auto_convert_format = auto_convert_format

        self.max_entities = max_entities
        self.max_relations = max_relations
        self.entity_embeddings: Dict[str, list] = {}
        self.context_embeddings: Dict[str, list] = {}
        self.entity_embedding_relation: Dict[str, Set] = collections.defaultdict(set)

        logger.info(f"Initializing {name} dataset ({version} version)")
        logger.info(f"Language: {language}")
        logger.info(f"Data will be stored in: {self.dataset_path}")

        self.dataset_path.mkdir(parents=True, exist_ok=True)

        self.train_val_news_data: Dict[str, keras.KerasTensor] = {}
        self.train_behaviors_data: Dict[str, keras.KerasTensor] = {}
        self.val_behaviors_data: Dict[str, keras.KerasTensor] = {}
        self.test_news_data: Dict[str, keras.KerasTensor] = {}
        self.test_behaviors_data: Dict[str, keras.KerasTensor] = {}

        np.random.seed(seed)

        # Initialize news ID mapping for string-based IDs
        self._news_id_to_int_map = {}
        self._int_to_news_id_map = {}  # Inverse mapping for fast lookup
        self._next_news_int_id = 0

        # Handle format conversion before processing news (if needed)
        if self._check_conversion_needed():
            logger.info("Format conversion needed. Converting custom format to MIND format...")
            conversion_path = self._convert_custom_format()
            logger.info(f"Conversion dataset saved at: {conversion_path}")

        # Handle auto-splitting if needed
        if self.auto_split_behaviors and not self._has_pre_split_data() and self._has_single_behaviors_file():
            logger.info("Auto-splitting enabled and single behaviors.tsv found. Performing automatic split...")
            self._auto_split_behaviors_file()

        self.processed_news = self.process_news()
        self._load_data(mode)

    @property
    def train_size(self) -> int:
        return (
            len(self.train_behaviors_data["impression_ids"])
            if "impression_ids" in self.train_behaviors_data
            else 0
        )

    @property
    def val_size(self) -> int:
        return (
            len(self.val_behaviors_data["impression_ids"])
            if "impression_ids" in self.val_behaviors_data
            else 0
        )

    @property
    def test_size(self) -> int:
        return (
            len(self.test_behaviors_data["impression_ids"])
            if "impression_ids" in self.test_behaviors_data
            else 0
        )

    def parse_news_id(self, news_id: str) -> int:
        """Parse news ID to integer, creating mapping for string IDs if needed."""
        if self.id_prefix and news_id.startswith(self.id_prefix):
            try:
                return int(news_id.split(self.id_prefix)[1])
            except ValueError:
                # String-based ID with prefix, map to integer
                if news_id not in self._news_id_to_int_map:
                    self._news_id_to_int_map[news_id] = self._next_news_int_id
                    self._int_to_news_id_map[self._next_news_int_id] = news_id  # Store inverse mapping
                    self._next_news_int_id += 1
                return self._news_id_to_int_map[news_id]

        try:
            return int(news_id)
        except ValueError:
            # String-based ID without prefix, map to integer
            if news_id not in self._news_id_to_int_map:
                self._news_id_to_int_map[news_id] = self._next_news_int_id
                self._int_to_news_id_map[self._next_news_int_id] = news_id  # Store inverse mapping
                self._next_news_int_id += 1
            return self._news_id_to_int_map[news_id]

    def get_int_to_news_id_map(self) -> Dict[int, str]:
        """Get the inverse mapping from integer IDs to string news IDs."""
        return self._int_to_news_id_map

    def _rebuild_id_mappings(self) -> None:
        """Rebuild ID mappings from processed news data when loading existing files."""
        if "news_ids_original_strings" not in self.processed_news:
            logger.warning("No news_ids_original_strings found in processed news data")
            return

        logger.info("Rebuilding ID mappings from processed news data...")
        news_ids = self.processed_news["news_ids_original_strings"]

        # Clear existing mappings
        self._news_id_to_int_map.clear()
        self._int_to_news_id_map.clear()
        self._next_news_int_id = 0  # Reset the counter

        # Rebuild mappings by parsing each news ID
        for original_id in news_ids:
            self.parse_news_id(original_id)
            # The parse_news_id method will populate the mappings automatically

        logger.info(f"Rebuilt ID mappings: {len(self._news_id_to_int_map)} news IDs mapped")

    def parse_user_id(self, user_id: str) -> int:
        """Parse user ID to integer, handling optional prefix."""
        if self.user_id_prefix and user_id.startswith(self.user_id_prefix):
            return int(user_id.split(self.user_id_prefix)[1])
        return int(user_id)

    def _has_pre_split_data(self) -> bool:
        """Check if dataset has pre-split train/valid directories with behaviors.tsv files."""
        train_behaviors = self.dataset_path / "train" / "behaviors.tsv"
        valid_behaviors = self.dataset_path / "valid" / "behaviors.tsv"
        return train_behaviors.exists() and valid_behaviors.exists()

    def _has_single_behaviors_file(self) -> bool:
        """Check if dataset has a single behaviors.tsv file in the root directory."""
        root_behaviors = self.dataset_path / "behaviors.tsv"
        return root_behaviors.exists()

    def _convert_custom_format(self) -> Path:
        """Convert custom dataset format to MIND format and return conversion path."""
        logger.info("Auto-converting custom dataset format to MIND format...")

        # Create conversion directory
        conversion_path = self.dataset_path / "conversion"

        # Check if conversion already exists
        conversion_behaviors = conversion_path / "behaviors.tsv"
        conversion_news = conversion_path / "news.tsv"

        if conversion_behaviors.exists() and conversion_news.exists():
            logger.info(f"Conversion already exists at {conversion_path}, skipping conversion")
            return conversion_path

        # Perform conversion
        try:
            preprocess_custom_dataset(
                input_dir=self.dataset_path,
                output_dir=conversion_path,
                user_id_prefix=self.user_id_prefix,
                news_id_prefix=self.id_prefix,
                time_format="mind"
            )
            logger.info(f"Successfully converted dataset to MIND format at {conversion_path}")
            return conversion_path
        except Exception as e:
            logger.error(f"Failed to convert dataset format: {e}")
            raise RuntimeError(f"Dataset format conversion failed: {e}")

    def _check_conversion_needed(self) -> bool:
        """Check if custom format conversion is needed."""
        if not self.auto_convert_format:
            logger.debug("auto_convert_format is False, no conversion needed")
            return False

        # Check if original files exist but conversion doesn't
        original_behaviors = self.dataset_path / "behaviors.tsv"
        original_news = self.dataset_path / "news.tsv"
        conversion_path = self.dataset_path / "conversion"

        has_original_files = original_behaviors.exists() and original_news.exists()
        has_conversion = (conversion_path / "behaviors.tsv").exists()

        # Convert if we have original files but no pre-split data and no conversion yet
        needs_conversion = has_original_files and not has_conversion
        logger.debug(f"needs_conversion={needs_conversion}")
        return needs_conversion

    def _auto_split_behaviors_file(self) -> None:
        """Automatically split a single behaviors.tsv file into train/valid/test splits."""

        logger.info("Auto-splitting single behaviors.tsv file into train/valid/test splits...")

        if self.auto_convert_format:
            root_behaviors_path = self.dataset_path / "conversion" / "behaviors.tsv"
        else:
            # In case no conversion is needed
            root_behaviors_path = self.dataset_path / "behaviors.tsv"

        if not root_behaviors_path.exists():
            raise FileNotFoundError(f"behaviors.tsv not found at {root_behaviors_path}")

        behaviors_df = pd.read_csv(
            root_behaviors_path,
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        behaviors_df["time"] = pd.to_datetime(behaviors_df['time'], format='%m/%d/%Y %I:%M:%S %p')
        behaviors_df = behaviors_df.sort_values("time").reset_index(drop=True)

        unique_dates = sorted(behaviors_df["time"].dt.date.unique())

        test_start_date = unique_dates[-1]  # Get last day

        test_behaviors = behaviors_df[behaviors_df["time"].dt.date >= test_start_date]
        remaining_behaviors = behaviors_df[behaviors_df["time"].dt.date < test_start_date]

        logger.info(f"Test split: {len(test_behaviors):,} behaviors from {test_start_date} onwards")

        train_dir = self.dataset_path / "train"
        valid_dir = self.dataset_path / "valid"  # This serves as test data per existing design

        train_dir.mkdir(exist_ok=True)
        valid_dir.mkdir(exist_ok=True)

        remaining_behaviors.to_csv(
            train_dir / "behaviors.tsv",
            sep="\t",
            header=False,
            index=False
        )

        test_behaviors.to_csv(
            valid_dir / "behaviors.tsv",
            sep="\t",
            header=False,
            index=False
        )

        if self.auto_convert_format:
            news_path = self.dataset_path / "conversion" / "news.tsv"
        else:
            # In case no conversion is needed
            news_path = self.dataset_path / "news.tsv"

        if news_path.exists():
            shutil.copy2(news_path, train_dir / "news.tsv")
            shutil.copy2(news_path, valid_dir / "news.tsv")
            logger.info("Copied news.tsv to train and valid directories")

        logger.info("Successfully auto-split behaviors.tsv: train+val data -> train/, test data -> valid/")

    def process_news(self) -> Dict[str, np.ndarray]:
        """Process news articles into numerical format, building vocabulary and tokenizing titles."""
        processed_path = self.dataset_path / "processed"
        os.makedirs(processed_path, exist_ok=True)

        vocab_file = processed_path / f"vocab_thresh{self.word_threshold}_{self.language}.pkl"
        processed_news_file = processed_path / f"processed_news_thresh{self.word_threshold}_{self.language}.pkl"
        embeddings_file = processed_path / f"filtered_embeddings_thresh{self.word_threshold}_{self.language}.npy"

        if self.download_if_missing:
            self.download_dataset()

        if not (processed_news_file.exists() and embeddings_file.exists() and vocab_file.exists()):
            logger.info(f"Processing news data with word_threshold={self.word_threshold}, language={self.language}...")

            # Read all news data
            all_news_df = self._read_all_news()

            if self.use_knowledge_graph:
                self._process_knowledge_graph(all_news_df)

            unique_news_ids_str = all_news_df["id"].unique()
            self.news_str_id_to_int_idx = {
                nid_str: i for i, nid_str in enumerate(unique_news_ids_str)
            }

            # Create category mappings
            unique_categories = sorted(all_news_df["category"].unique())
            unique_subcategories = sorted(all_news_df["subcategory"].unique())
            self.category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
            self.subcategory_to_idx = {subcat: idx for idx, subcat in enumerate(unique_subcategories)}

            category_indices = np.zeros(len(unique_news_ids_str), dtype=np.int32)
            subcategory_indices = np.zeros(len(unique_news_ids_str), dtype=np.int32)

            for nid_str, cat, subcat in zip(
                    all_news_df["id"], all_news_df["category"], all_news_df["subcategory"]
            ):
                int_idx = self.news_str_id_to_int_idx[nid_str]
                category_indices[int_idx] = self.category_to_idx[cat]
                subcategory_indices[int_idx] = self.subcategory_to_idx[subcat]

            # Build vocabulary
            self.vocab = self._build_vocabulary(all_news_df)
            with open(vocab_file, "wb") as f:
                pickle.dump(self.vocab, f)

            # Tokenize all news
            tokenized_titles_np, tokenized_abstracts_np = self._tokenize_all_news(all_news_df)

            # Create embeddings
            embedding_matrix = self._create_embeddings()
            np.save(embeddings_file, embedding_matrix)

            processed_news_content: Dict[str, Any] = {
                "news_ids_original_strings": unique_news_ids_str,
                "tokens": tokenized_titles_np,
                "abstract_tokens": tokenized_abstracts_np,
                "vocab_size": len(self.vocab),
                "category_indices": category_indices,
                "subcategory_indices": subcategory_indices,
                "num_categories": len(unique_categories),
                "num_subcategories": len(unique_subcategories),
            }

            with open(processed_news_file, "wb") as f:
                pickle.dump(processed_news_content, f)
            logger.info("Finished processing news data.")

        else:
            logger.info(f"Loading processed news data from cache...")
            with open(vocab_file, "rb") as f:
                self.vocab = pickle.load(f)
            with open(processed_news_file, "rb") as f:
                processed_news_content = pickle.load(f)

            self.news_str_id_to_int_idx = {
                nid_str: i
                for i, nid_str in enumerate(processed_news_content["news_ids_original_strings"])
            }

        logger.info(f"Loading filtered embeddings...")
        embedding_matrix_loaded = np.load(embeddings_file)
        processed_news_content["embeddings"] = keras.ops.cast(
            keras.ops.convert_to_tensor(embedding_matrix_loaded), self.float_dtype
        )

        self.news_id_str_to_tokens: Dict[str, np.ndarray] = {
            nid_str: processed_news_content["tokens"][self.news_str_id_to_int_idx[nid_str]]
            for nid_str in processed_news_content["news_ids_original_strings"]
        }
        processed_news_content["vocab"] = self.vocab

        return processed_news_content

    def _read_all_news(self) -> pd.DataFrame:
        """Read and combine all news data from train/valid/test splits or single file."""
        logger.info("Reading all news (train, dev, test) for vocab building...")
        news_dfs = []

        def read_news_file(file_path: Path) -> pd.DataFrame:
            """Read a news.tsv file and auto-detect column format."""

            # Column names
            column_names = [
                "id", "category", "subcategory", "title", "abstract",
                "url", "title_entities", "abstract_entities"
            ]

            news = pd.read_table(
                file_path,
                header=None,
                names=column_names,
                na_filter=False,
                on_bad_lines='skip',  # Skip malformed lines
                engine='python'  # Use Python engine for better error handling
            )

            return news

        # Otherwise look for split files
        train_news_path = self.dataset_path / "train" / "news.tsv"
        if train_news_path.exists():
            news_dfs.append(read_news_file(train_news_path))

        valid_news_path = self.dataset_path / "valid" / "news.tsv"
        if valid_news_path.exists():
            news_dfs.append(read_news_file(valid_news_path))

        test_news_path = self.dataset_path / "test" / "news.tsv"
        if test_news_path.exists():
            news_dfs.append(read_news_file(test_news_path))

        if not news_dfs:
            raise FileNotFoundError(
                f"No news.tsv files found in {self.dataset_path}. "
                f"Expected either a single news.tsv file or news.tsv files in train/valid/test directories."
            )

        all_news_df = pd.concat(news_dfs, ignore_index=True).drop_duplicates(
            subset=["id"], keep="first"
        )
        logger.info("All news dataframes joined and duplicates dropped...")
        return all_news_df

    def _build_vocabulary(self, news_df: pd.DataFrame) -> Dict[str, int]:
        """Build vocabulary from training news titles and abstracts.
        
        Uses word_threshold to filter out rare words (appearing < threshold times),
        reducing vocabulary size and improving generalization by avoiding overfitting to typos/outliers.
        """
        logger.info("Building vocabulary from news titles and abstracts...")
        word_counter = collections.Counter()

        # Count words from both titles and abstracts
        total_items = len(news_df) * 2  # titles + abstracts
        logger.info(f"Counting words from {len(news_df):,} news titles and abstracts...")

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
        ) as progress:
            task = progress.add_task("Counting words in titles and abstracts...", total=total_items)

            # Process titles
            for title_text in news_df["title"].values:
                words = self._segment_text_into_words(str(title_text))
                for word in words:
                    processed_word = "<NUM>" if string_is_number(word) else word
                    word_counter[processed_word] += 1
                progress.advance(task)

            # Process abstracts
            for abstract_text in news_df["abstract"].values:
                words = self._segment_text_into_words(str(abstract_text))
                for word in words:
                    processed_word = "<NUM>" if string_is_number(word) else word
                    word_counter[processed_word] += 1
                progress.advance(task)

        # Build vocabulary
        vocab = {"[PAD]": 0, "[UNK]": 1}
        token_id_counter = 2

        if "<NUM>" in word_counter and word_counter["<NUM>"] >= self.word_threshold:
            vocab["<NUM>"] = token_id_counter
            token_id_counter += 1

        sorted_word_counts = sorted(
            word_counter.items(), key=lambda item: item[1], reverse=True
        )

        for word, count in sorted_word_counts:
            if word in vocab:
                continue
            if count >= self.word_threshold:
                vocab[word] = token_id_counter
                token_id_counter += 1

        logger.info(f"Vocabulary size (after threshold {self.word_threshold}): {len(vocab)}")
        return vocab

    def _tokenize_all_news(self, all_news_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenize all news titles and abstracts."""
        logger.info("Tokenizing all news titles and abstracts...")
        num_unique_news = len(all_news_df["id"].unique())

        tokenized_titles_np = np.full(
            (num_unique_news, self.max_title_length), self.vocab["[PAD]"], dtype=np.int32
        )
        tokenized_abstracts_np = np.full(
            (num_unique_news, self.max_abstract_length), self.vocab["[PAD]"], dtype=np.int32
        )

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
        ) as progress:
            task = progress.add_task(
                "Tokenizing all news titles and abstracts...", total=len(all_news_df)
            )
            for nid_str, title_text, abstract_text in zip(
                    all_news_df["id"], all_news_df["title"], all_news_df["abstract"]
            ):
                int_idx = self.news_str_id_to_int_idx[nid_str]
                tokenized_titles_np[int_idx] = self.tokenize_text(
                    str(title_text),
                    self.vocab,
                    self.max_title_length,
                    unk_token_id=self.vocab["[UNK]"],
                    pad_token_id=self.vocab["[PAD]"],
                )
                tokenized_abstracts_np[int_idx] = self.tokenize_text(
                    str(abstract_text),
                    self.vocab,
                    self.max_abstract_length,
                    unk_token_id=self.vocab["[UNK]"],
                    pad_token_id=self.vocab["[PAD]"],
                )
                progress.advance(task)

        return tokenized_titles_np, tokenized_abstracts_np

    def _create_embeddings(self) -> np.ndarray:
        """Create embedding matrix based on language and embedding type."""
        logger.info(f"Creating embeddings for language: {self.language}, type: {self.embedding_type}...")

        if self.embedding_type == "glove" and self.language == "english":
            return self._create_glove_embeddings()
        elif self.embedding_type == "bpemb":
            return self._create_bpemb_embeddings()
        else:
            # NOT IDEAl
            return self._create_random_embeddings()

    def _create_glove_embeddings(self) -> np.ndarray:
        """Create embedding matrix using GloVe embeddings."""
        glove_tensor_tf, glove_vocab_map = (
            self.embeddings_manager.load_glove_embeddings_tf_and_vocab_map(self.embedding_size)
        )
        if glove_tensor_tf is None or glove_vocab_map is None:
            raise ValueError("GloVe embeddings or vocab map could not be loaded.")

        glove_array = keras.ops.convert_to_numpy(glove_tensor_tf)
        glove_mean_np = np.mean(glove_array, axis=0)
        glove_std_np = np.std(glove_array, axis=0)

        embedding_matrix = np.zeros((len(self.vocab), self.embedding_size), dtype=np.float32)
        embedding_matrix[self.vocab["[PAD]"]] = np.zeros(self.embedding_size, dtype=np.float32)
        embedding_matrix[self.vocab["[UNK]"]] = np.random.normal(
            loc=glove_mean_np, scale=glove_std_np, size=self.embedding_size
        ).astype(np.float32)

        if "<NUM>" in self.vocab:
            num_token_id = self.vocab["<NUM>"]
            glove_num_idx = glove_vocab_map.get("<NUM>")
            if glove_num_idx is not None:
                embedding_matrix[num_token_id] = glove_array[glove_num_idx]
            else:
                glove_number_idx = glove_vocab_map.get("number")
                if glove_number_idx is not None:
                    embedding_matrix[num_token_id] = glove_array[glove_number_idx]
                else:
                    embedding_matrix[num_token_id] = np.random.normal(
                        loc=glove_mean_np, scale=glove_std_np, size=self.embedding_size
                    ).astype(np.float32)

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
        ) as progress:
            task = progress.add_task("Populating embedding matrix...", total=len(self.vocab))
            for word, idx in self.vocab.items():
                if word in ["[PAD]", "[UNK]", "<NUM>"]:
                    progress.advance(task)
                    continue
                glove_word_idx = glove_vocab_map.get(word)
                if glove_word_idx is not None:
                    embedding_matrix[idx] = glove_array[glove_word_idx]
                else:
                    embedding_matrix[idx] = np.random.normal(
                        loc=glove_mean_np, scale=glove_std_np, size=self.embedding_size
                    ).astype(np.float32)
                progress.advance(task)

        return embedding_matrix

    def _create_bpemb_embeddings(self) -> np.ndarray:
        """Create embedding matrix using BPEmb pre-trained embeddings."""
        logger.info(f"Creating BPEmb embeddings for language: {self.language}")

        # Map language names to BPEmb language codes
        lang_map = {
            "japanese": "ja",
            "german": "de",
            "french": "fr",
            "spanish": "es",
            "italian": "it",
            "portuguese": "pt",
            "russian": "ru",
            "korean": "ko",
            "chinese": "zh",
            "arabic": "ar",
            "hindi": "hi",
            "turkish": "tr",
            "polish": "pl",
            "dutch": "nl",
            "english": "en"
        }

        lang_code = lang_map.get(self.language.lower(), self.language.lower())

        try:
            # Load BPEmb embeddings using embeddings manager
            logger.info(f"Loading BPEmb embeddings for language: {lang_code}")
            bpemb_embeddings = self.embeddings_manager.get_bpemb_embeddings(
                language=lang_code,
                vocab_size=200000,
                dim=self.embedding_size
            )

            if not bpemb_embeddings:
                logger.warning(f"No BPEmb embeddings loaded for {lang_code}")
                return self._create_random_embeddings()

            # Create embedding matrix aligned with our vocabulary
            logger.info(f"Creating embedding matrix from {len(bpemb_embeddings):,} BPE tokens")
            embedding_matrix = np.random.randn(len(self.vocab), self.embedding_size).astype(np.float32) * 0.1

            # Set PAD token to zeros
            embedding_matrix[self.vocab["[PAD]"]] = np.zeros(self.embedding_size, dtype=np.float32)

            # Map our vocabulary words to BPE tokens and use their embeddings
            matched_words = 0
            for word, idx in self.vocab.items():
                if word in ["[PAD]", "[UNK]", "<NUM>"]:
                    continue

                # Try exact match first
                if word in bpemb_embeddings:
                    embedding_matrix[idx] = bpemb_embeddings[word]
                    matched_words += 1
                # For words not in BPE vocab, try lowercase
                elif word.lower() in bpemb_embeddings:
                    embedding_matrix[idx] = bpemb_embeddings[word.lower()]
                    matched_words += 1
                # For compound words, try to average subword embeddings
                else:
                    # Find BPE tokens that are substrings of the word
                    subword_embeddings = []
                    for bpe_token in bpemb_embeddings:
                        if bpe_token in word and len(bpe_token) > 1:
                            subword_embeddings.append(bpemb_embeddings[bpe_token])

                    if subword_embeddings:
                        # Average the subword embeddings
                        embedding_matrix[idx] = np.mean(subword_embeddings, axis=0)
                        matched_words += 1

            match_percentage = (matched_words / len(self.vocab)) * 100
            logger.info(f"Successfully created BPEmb embedding matrix: {embedding_matrix.shape}")
            logger.info(f"Matched {matched_words}/{len(self.vocab)} words ({match_percentage:.1f}%)")

            return embedding_matrix

        except Exception as e:
            logger.error(f"Failed to load BPEmb embeddings for {lang_code}: {e}")
            logger.warning("Falling back to random embeddings")
            return self._create_random_embeddings()

    def _create_random_embeddings(self) -> np.ndarray:
        """Create random embedding matrix for non-English languages."""
        logger.info(f"Creating random embeddings for language: {self.language}")
        embedding_matrix = np.random.randn(len(self.vocab), self.embedding_size).astype(np.float32) * 0.1
        embedding_matrix[self.vocab["[PAD]"]] = np.zeros(self.embedding_size, dtype=np.float32)
        return embedding_matrix

    def _load_data(self, mode: str = "train") -> bool:
        """Try to load processed tensor data from disk."""

        processed_path = self.dataset_path / "processed"
        files_exist = (
                (processed_path / "processed_train.pkl").exists()
                and (processed_path / "processed_val.pkl").exists()
                and (processed_path / "processed_test.pkl").exists()
        )

        if not files_exist:
            self._process_data()
        else:
            # If files exist, we need to rebuild ID mappings since they're only created during processing
            self._rebuild_id_mappings()

        logger.info("Files have already been processed, loading data...")
        try:
            if mode == "train":
                logger.info("Loading train behaviors data...")
                self.train_behaviors_data = pd.read_pickle(processed_path / "processed_train.pkl")
                logger.info("Loading validation behaviors data...")
                self.val_behaviors_data = pd.read_pickle(processed_path / "processed_val.pkl")

                if self.data_fraction_train < 1.0:
                    self.train_behaviors_data = apply_data_fraction(
                        self.train_behaviors_data, self.data_fraction_train
                    )

                if self.data_fraction_val < 1.0:
                    self.val_behaviors_data = apply_data_fraction(
                        self.val_behaviors_data, self.data_fraction_val
                    )

                self._display_statistics(
                    mode,
                    processed_news=self.processed_news,
                    train_behaviors_data=self.train_behaviors_data,
                    val_behaviors_data=self.val_behaviors_data,
                )

                logger.info("Successfully loaded processed train/val tensors")
                return True

            else:
                logger.info("Loading test behaviors data...")
                self.test_behaviors_data = pd.read_pickle(processed_path / "processed_test.pkl")

                if self.data_fraction_test < 1.0:
                    self.test_behaviors_data = apply_data_fraction(
                        self.test_behaviors_data, self.data_fraction_test
                    )

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

        logger.info("Processing train data...")
        train_behaviors_dict, val_behaviors_dict = self.get_train_val_data()

        logger.info("Saving training data...")
        with open(processed_path / "processed_train.pkl", "wb") as f:
            pickle.dump(train_behaviors_dict, f)

        logger.info("Processing validation data...")
        with open(processed_path / "processed_val.pkl", "wb") as f:
            pickle.dump(val_behaviors_dict, f)

        logger.info("Processing test data...")
        test_behaviors_dict = self.get_test_data()

        logger.info("Saving test data...")
        with open(processed_path / "processed_test.pkl", "wb") as f:
            pickle.dump(test_behaviors_dict, f)

        self.train_behaviors_data = train_behaviors_dict
        self.val_behaviors_data = val_behaviors_dict
        self.test_behaviors_data = test_behaviors_dict

        logger.info("Generating dataset summary...")
        self.generate_dataset_summary()

        logger.info("Preprocessing complete!")

    def download_dataset(self) -> None:
        """Download and extract dataset if not already present"""
        if self.dataset_path.exists() and any(self.dataset_path.iterdir()):
            logger.info(f"Found existing dataset at {self.dataset_path}")
            return

        if not self.urls:
            logger.warning(f"No URLs provided for downloading dataset. Assuming data exists at {self.dataset_path}")
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

                    progress.add_task(f"Extracting {split} set...", total=100)

                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_path)

                    zip_path.unlink()
                    logger.info(f"Successfully processed {split} set")
                else:
                    logger.info(f"Found existing {split} set at {extract_path}")

        if self.use_knowledge_graph:
            self._download_knowledge_graph()

        self.cache_manager.add_to_cache(
            self.name.lower().replace(" ", "_"),
            self.version,
            "dataset",
            metadata={
                "splits": list(self.urls.keys()) if self.urls else [],
                "max_title_length": self.max_title_length,
                "max_history_length": self.max_history_length,
                "version": self.version,
                "use_knowledge_graph": self.use_knowledge_graph,
                "language": self.language,
            },
        )

    def _download_knowledge_graph(self) -> None:
        """Download and extract knowledge graph data."""
        graph_extract_path = self.dataset_path / "download" / "wikidata-graph"

        if not graph_extract_path.exists():
            graph_url = (
                "https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip"
            )
            graph_zip_path = self.dataset_path / "download" / "wikidata-graph.zip"
            self._download_and_unzip_file(
                graph_url, graph_zip_path, graph_extract_path, "knowledge graph"
            )
        else:
            logger.info("Found existing knowledge graph data")

    def _download_and_unzip_file(
            self, url: str, zip_path: Path, extract_path: Path, description: str
    ) -> None:
        """Downloads a file from a URL and unzips it."""
        logger.info(f"Downloading {description} data from {url} to {zip_path}...")
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        urllib.request.urlretrieve(url, zip_path)

        logger.info(f"Extracting {description} to {extract_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        zip_path.unlink()
        logger.info(f"Successfully downloaded and extracted {description} data.")

    def process_behaviors(
            self, behaviors_df: pd.DataFrame, stage: str
    ) -> Dict[str, Union[np.ndarray, list]]:
        """Process behaviors storing only tokens, not embeddings."""
        histories_news_ids: List[list] = []
        history_news_tokens: List[list] = []
        history_news_abstract_tokens: List[list] = []
        history_news_categories: List[list] = []
        history_news_subcategories: List[list] = []
        candidate_news_ids: List[list] = []
        candidate_news_tokens: List[list] = []
        candidate_news_abstract_tokens: List[list] = []
        candidate_news_categories: List[list] = []
        candidate_news_subcategories: List[list] = []
        labels: List[list] = []
        impression_ids: List[int] = []
        user_ids: List[str] = []

        total_original_rows = len(behaviors_df)
        total_positives = 0
        rows_with_multiple_positives = 0
        max_positives_in_row = 0

        # tokens dictionaries
        news_tokens: Dict[int, np.ndarray] = dict(
            zip(
                [self.parse_news_id(nid) for nid in self.processed_news["news_ids_original_strings"]],
                self.processed_news["tokens"],
            )
        )

        news_abstract_tokens: Dict[int, np.ndarray] = dict(
            zip(
                [self.parse_news_id(nid) for nid in self.processed_news["news_ids_original_strings"]],
                self.processed_news["abstract_tokens"],
            )
        )

        news_categories: Dict[int, int] = dict(
            zip(
                [self.parse_news_id(nid) for nid in self.processed_news["news_ids_original_strings"]],
                self.processed_news["category_indices"],
            )
        )
        news_subcategories: Dict[int, int] = dict(
            zip(
                [self.parse_news_id(nid) for nid in self.processed_news["news_ids_original_strings"]],
                self.processed_news["subcategory_indices"],
            )
        )

        # Filter out behaviors that don't have at least one positive and negative samples
        initial_count = len(behaviors_df)

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
        ) as progress:
            filter_task = progress.add_task(
                "Filtering behaviors without at least one negative sample...",
                total=len(behaviors_df)
            )

            def progress_callback():
                progress.advance(filter_task)

            behaviors_mask = behaviors_df['impressions'].apply(
                lambda impressions_item: _has_atleast_one_pos_neg_samples(
                    impressions_item,
                    parse_news_id=self.parse_news_id,
                    available_news_ids=set(news_tokens.keys()),
                    progress_callback=progress_callback
                )
            )

        behaviors_df = behaviors_df[behaviors_mask].copy()

        n_removed = initial_count - len(behaviors_df)
        logger.info(f"Removed {n_removed} training behaviors without both positive and negative samples")
        logger.info(f"Remaining training behaviors: {len(behaviors_df)}")

        if n_removed > 0:
            removal_percentage = (n_removed / initial_count) * 100
            logger.info(f"Filtered out {removal_percentage:.1f}% of training behaviors")

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
                impressions = str(row["impressions"]).split()
                user_id = self.parse_user_id(str(row["user_id"]))

                positives_count = sum(imp.split("-")[1] == "1" for imp in impressions)

                total_positives += positives_count
                if positives_count > 1:
                    rows_with_multiple_positives += 1
                max_positives_in_row = max(max_positives_in_row, positives_count)

                history = str(row["history"]).split() if pd.notna(row["history"]) else []
                history = history[-self.max_history_length:]  # filter out max length

                # Parse news IDs and filter out missing ones
                history_nid_list = []
                for h in history:
                    h_idx = self.parse_news_id(h)
                    if h_idx in news_tokens:  # Only include if news exists
                        history_nid_list.append(h_idx)

                # Get all the tokens for (title + abstract + category + subcategory)
                curr_history_tokens = [news_tokens[h_idx] for h_idx in history_nid_list]
                curr_history_abstract_tokens = [
                    news_abstract_tokens[h_idx] for h_idx in history_nid_list
                ]
                curr_history_categories = [news_categories[h_idx] for h_idx in history_nid_list]
                curr_history_subcategories = [
                    news_subcategories[h_idx] for h_idx in history_nid_list
                ]

                # padding
                history_pad_length = self.max_history_length - len(history_nid_list)
                history_nid_list = [0] * history_pad_length + history_nid_list
                curr_history_tokens = [[0] * self.max_title_length] * history_pad_length + curr_history_tokens
                curr_history_abstract_tokens = [[0] * self.max_abstract_length] * history_pad_length + curr_history_abstract_tokens
                curr_history_categories = [0] * history_pad_length + curr_history_categories
                curr_history_subcategories = [0] * history_pad_length + curr_history_subcategories

                # sampling candidate news
                cand_nid_group_list, label_group_list = self.sampler.sample_candidates_news(
                    stage=stage,
                    candidates=impressions,
                    random_train_samples=self.random_train_samples,
                    parse_news_id=self.parse_news_id,
                    available_news_ids=set(news_tokens.keys()),
                )

                if stage == "train":
                    for cand_nid_group, label_group in zip(cand_nid_group_list, label_group_list):
                        histories_news_ids.append(history_nid_list)
                        history_news_tokens.append(curr_history_tokens)
                        history_news_abstract_tokens.append(curr_history_abstract_tokens)
                        history_news_categories.append(curr_history_categories)
                        history_news_subcategories.append(curr_history_subcategories)
                        candidate_news_ids.append(cand_nid_group)
                        candidate_news_tokens.append([news_tokens[nid] for nid in cand_nid_group])
                        candidate_news_abstract_tokens.append(
                            [news_abstract_tokens[nid] for nid in cand_nid_group]
                        )
                        candidate_news_categories.append(
                            [news_categories[nid] for nid in cand_nid_group]
                        )
                        candidate_news_subcategories.append(
                            [news_subcategories[nid] for nid in cand_nid_group]
                        )
                        labels.append(label_group)
                        impression_ids.append(row["impression_id"])
                        user_ids.append(user_id)
                else:
                    cand_nid_group, label_group = cand_nid_group_list, label_group_list

                    histories_news_ids.append(history_nid_list)
                    history_news_tokens.append(curr_history_tokens)
                    history_news_abstract_tokens.append(curr_history_abstract_tokens)
                    history_news_categories.append(curr_history_categories)
                    history_news_subcategories.append(curr_history_subcategories)
                    candidate_news_ids.append(cand_nid_group)
                    candidate_news_tokens.append([news_tokens[nid] for nid in cand_nid_group])
                    candidate_news_abstract_tokens.append(
                        [news_abstract_tokens[nid] for nid in cand_nid_group]
                    )
                    candidate_news_categories.append(
                        [news_categories[nid] for nid in cand_nid_group]
                    )
                    candidate_news_subcategories.append(
                        [news_subcategories[nid] for nid in cand_nid_group]
                    )
                    labels.append(label_group)
                    impression_ids.append(row["impression_id"])
                    user_ids.append(user_id)

                progress.advance(task)

            if stage == "train":
                result: Dict[str, Union[np.ndarray, list]] = {
                    "histories_news_ids": np.array(histories_news_ids, dtype=np.int32),
                    "history_news_tokens": np.array(history_news_tokens, dtype=np.int32),
                    "history_news_abstract_tokens": np.array(
                        history_news_abstract_tokens, dtype=np.int32
                    ),
                    "history_news_categories": np.array(history_news_categories, dtype=np.int32),
                    "history_news_subcategories": np.array(
                        history_news_subcategories, dtype=np.int32
                    ),
                    "candidate_news_ids": np.array(candidate_news_ids, dtype=np.int32),
                    "candidate_news_tokens": np.array(candidate_news_tokens, dtype=np.int32),
                    "candidate_news_abstract_tokens": np.array(
                        candidate_news_abstract_tokens, dtype=np.int32
                    ),
                    "candidate_news_categories": np.array(
                        candidate_news_categories, dtype=np.int32
                    ),
                    "candidate_news_subcategories": np.array(
                        candidate_news_subcategories, dtype=np.int32
                    ),
                    "labels": np.array(labels, dtype=np.float32 if self.float_dtype in ["float32",
                                                                                        "mixed_float16"] else np.float16),
                    "impression_ids": np.array(impression_ids, dtype=np.int32),
                    "user_ids": np.array(user_ids, dtype=np.int32),
                }
            else:
                result = {
                    "histories_news_ids": histories_news_ids,
                    "history_news_tokens": history_news_tokens,
                    "history_news_abstract_tokens": history_news_abstract_tokens,
                    "history_news_categories": history_news_categories,
                    "history_news_subcategories": history_news_subcategories,
                    "candidate_news_ids": candidate_news_ids,
                    "candidate_news_tokens": candidate_news_tokens,
                    "candidate_news_abstract_tokens": candidate_news_abstract_tokens,
                    "candidate_news_categories": candidate_news_categories,
                    "candidate_news_subcategories": candidate_news_subcategories,
                    "labels": labels,
                    "impression_ids": impression_ids,
                    "user_ids": user_ids,
                }

            total_processed_rows = len(histories_news_ids)
            expansion_factor = total_processed_rows / total_original_rows

            logger.info(f"\nBehavior Processing Statistics ({stage}):")
            logger.info(f"Original number of rows: {total_original_rows:,}")
            logger.info(f"Processed number of rows: {total_processed_rows:,}")
            logger.info(f"Expansion factor: {expansion_factor:.2f}x")
            logger.info(f"Total positive samples: {total_positives:,}")
            logger.info(
                f"Rows with multiple positives: {rows_with_multiple_positives:,} "
                f"({rows_with_multiple_positives / total_original_rows * 100:.1f}%)"
            )
            logger.info(f"Maximum positives in a single row: {max_positives_in_row}")
            logger.info(f"Average positives per row: {total_positives / total_original_rows:.2f}")
            logger.info(f"Total users: {len(set(user_ids))}")

            return result

    def get_train_val_data(
            self,
            sampled_user_set: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, Union[np.ndarray, list]], Dict[str, Union[np.ndarray, list]]]:
        """Load and process training data, splitting into train and validation sets."""
        train_behaviors_path = self.dataset_path / "train" / "behaviors.tsv"

        if not train_behaviors_path.exists():
            raise FileNotFoundError(
                f"Training behaviors file not found at {train_behaviors_path}. "
                f"If using auto_split_behaviors=True, ensure the auto-split has been performed first."
            )

        behaviors_df = pd.read_csv(
            train_behaviors_path,
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        if sampled_user_set is not None:
            behaviors_df = behaviors_df[behaviors_df["user_id"].isin(list(sampled_user_set))]

        if self.validation_split_strategy == "random":
            logger.info(
                f"Using random split for validation: {self.validation_split_percentage * 100}% of training behaviors data, seed: {self.validation_split_seed}"
            )
            shuffled_df = behaviors_df.sample(
                frac=1, random_state=self.validation_split_seed
            ).reset_index(drop=True)

            val_size = int(len(shuffled_df) * self.validation_split_percentage)
            val_behaviors = shuffled_df.iloc[:val_size]
            train_behaviors = shuffled_df.iloc[val_size:]
            logger.info(
                f"Random split: Train size: {len(train_behaviors)}, Validation size: {len(val_behaviors)}"
            )

        elif self.validation_split_strategy == "chronological":
            logger.info(
                "Using chronological split for validation (last day of training behaviors data)."
            )
            behaviors_df["time"] = pd.to_datetime(behaviors_df["time"])
            last_day = behaviors_df["time"].max().date()
            train_behaviors = behaviors_df[behaviors_df["time"].dt.date < last_day]
            val_behaviors = behaviors_df[behaviors_df["time"].dt.date == last_day]
        else:
            raise ValueError(f"Unknown validation_split_strategy: {self.validation_split_strategy}")

        logger.info(f"Train behaviors: {len(train_behaviors):,}")
        train_behaviors_data = self.process_behaviors(train_behaviors, stage="train")

        logger.info(f"Validation behaviors: {len(val_behaviors):,}")
        val_behaviors_data = self.process_behaviors(val_behaviors, stage="val")

        return train_behaviors_data, val_behaviors_data

    def get_test_data(
            self,
            sampled_user_set: Optional[Set[str]] = None,
    ) -> Dict[str, Union[np.ndarray, list]]:
        """Load and process test data"""
        test_behaviors_path = self.dataset_path / "valid" / "behaviors.tsv"

        if not test_behaviors_path.exists():
            raise FileNotFoundError(
                f"Test behaviors file not found at {test_behaviors_path}. "
                f"If using auto_split_behaviors=True, ensure the auto-split has been performed first."
            )

        test_behaviors = pd.read_csv(
            test_behaviors_path,
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        if sampled_user_set is not None:
            test_behaviors = test_behaviors[test_behaviors["user_id"].isin(list(sampled_user_set))]

        logger.info(f"Test behaviors: {len(test_behaviors):,}")

        return self.process_behaviors(test_behaviors, stage="test")

    def train_dataloader(self, batch_size: int, model_name: str = "nrms"):
        """Create training dataset with token-based inputs."""
        return NewsDataLoader.create_train_dataset(
            history_news_tokens=self.train_behaviors_data["history_news_tokens"],
            history_news_abstract_tokens=self.train_behaviors_data["history_news_abstract_tokens"],
            history_news_category=self.train_behaviors_data["history_news_categories"],
            history_news_subcategory=self.train_behaviors_data["history_news_subcategories"],
            candidate_news_tokens=self.train_behaviors_data["candidate_news_tokens"],
            candidate_news_abstract_tokens=self.train_behaviors_data[
                "candidate_news_abstract_tokens"
            ],
            candidate_news_category=self.train_behaviors_data["candidate_news_categories"],
            candidate_news_subcategory=self.train_behaviors_data["candidate_news_subcategories"],
            labels=self.train_behaviors_data["labels"],
            user_ids=self.train_behaviors_data["user_ids"],
            batch_size=batch_size,
            process_title=self.process_title,
            process_abstract=self.process_abstract,
            process_category=self.process_category,
            process_subcategory=self.process_subcategory,
            process_user_id=self.process_user_id,
            model_name=model_name,
        )

    def user_history_dataloader(self, mode: str, batch_size: int) -> UserHistoryBatchDataloader:
        """Create dataloader for user history validation/testing."""
        if mode == "val":
            user_history_tokens = self.val_behaviors_data["history_news_tokens"]
            user_history_abstract_tokens = self.val_behaviors_data["history_news_abstract_tokens"]
            user_history_category = self.val_behaviors_data["history_news_categories"]
            user_history_subcategory = self.val_behaviors_data["history_news_subcategories"]
            impression_ids = self.val_behaviors_data["impression_ids"]
            user_ids = self.val_behaviors_data["user_ids"]
        elif mode == "test":
            user_history_tokens = self.test_behaviors_data["history_news_tokens"]
            user_history_abstract_tokens = self.test_behaviors_data["history_news_abstract_tokens"]
            user_history_category = self.test_behaviors_data["history_news_categories"]
            user_history_subcategory = self.test_behaviors_data["history_news_subcategories"]
            impression_ids = self.test_behaviors_data["impression_ids"]
            user_ids = self.test_behaviors_data["user_ids"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return UserHistoryBatchDataloader(
            history_tokens=user_history_tokens,
            history_abstract_tokens=user_history_abstract_tokens,
            history_category=user_history_category,
            history_subcategory=user_history_subcategory,
            impression_ids=impression_ids,
            user_ids=user_ids,
            batch_size=batch_size,
            process_title=self.process_title,
            process_abstract=self.process_abstract,
            process_category=self.process_category,
            process_subcategory=self.process_subcategory,
        )

    def impression_dataloader(self, mode: str) -> ImpressionIterator:
        """Create dataloader for impressions validation/testing."""
        if mode == "val":
            impression_tokens = self.val_behaviors_data["candidate_news_tokens"]
            impression_abstract_tokens = self.val_behaviors_data["candidate_news_abstract_tokens"]
            impression_category = self.val_behaviors_data["candidate_news_categories"]
            impression_subcategory = self.val_behaviors_data["candidate_news_subcategories"]
            labels = self.val_behaviors_data["labels"]
            impression_ids = self.val_behaviors_data["impression_ids"]
            candidate_ids = self.val_behaviors_data["candidate_news_ids"]
        elif mode == "test":
            impression_tokens = self.test_behaviors_data["candidate_news_tokens"]
            impression_abstract_tokens = self.test_behaviors_data["candidate_news_abstract_tokens"]
            impression_category = self.test_behaviors_data["candidate_news_categories"]
            impression_subcategory = self.test_behaviors_data["candidate_news_subcategories"]
            labels = self.test_behaviors_data["labels"]
            impression_ids = self.test_behaviors_data["impression_ids"]
            candidate_ids = self.test_behaviors_data["candidate_news_ids"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return ImpressionIterator(
            impression_tokens=impression_tokens,
            impression_abstract_tokens=impression_abstract_tokens,
            impression_category=impression_category,
            impression_subcategory=impression_subcategory,
            labels=labels,
            impression_ids=impression_ids,
            candidate_ids=candidate_ids,
            process_title=self.process_title,
            process_abstract=self.process_abstract,
            process_category=self.process_category,
            process_subcategory=self.process_subcategory,
        )

    def news_dataloader(self, batch_size: int) -> NewsBatchDataloader:
        """Create dataloader for processed news validation/testing."""
        news_ids = self.processed_news.get("news_ids_original_strings", np.array([]))
        news_tokens = self.processed_news.get("tokens", np.array([]))
        news_abstract_tokens = self.processed_news.get("abstract_tokens", np.array([]))
        news_category_indices = self.processed_news.get("category_indices", np.array([]))
        news_subcategory_indices = self.processed_news.get("subcategory_indices", np.array([]))

        return NewsBatchDataloader(
            news_ids=news_ids,
            news_tokens=keras.ops.convert_to_tensor(news_tokens),
            news_abstract_tokens=keras.ops.convert_to_tensor(news_abstract_tokens),
            news_category_indices=keras.ops.convert_to_tensor(news_category_indices),
            news_subcategory_indices=keras.ops.convert_to_tensor(news_subcategory_indices),
            batch_size=batch_size,
            process_title=self.process_title,
            process_abstract=self.process_abstract,
            process_category=self.process_category,
            process_subcategory=self.process_subcategory,
        )

    def _display_statistics(
            self,
            mode: str = "train",
            processed_news: Optional[Dict[str, Any]] = None,
            train_behaviors_data: Optional[Dict[str, Any]] = None,
            val_behaviors_data: Optional[Dict[str, Any]] = None,
            test_behaviors_data: Optional[Dict[str, Any]] = None,
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

    def _segment_text_into_words(self, sent: str) -> List[str]:
        """Segment a sentence string into a list of word strings."""
        pat = re.compile(r"[\w]+|[.,!?;|]")
        return pat.findall(sent.lower()) if isinstance(sent, str) else []

    def tokenize_text(
            self, text: str, vocab: Dict[str, int], max_len: int, unk_token_id: int, pad_token_id: int
    ) -> List[int]:
        """Converts a raw text string into a fixed-length sequence of numerical token IDs."""
        words = self._segment_text_into_words(text)
        tokens = []
        for word in words:
            processed_word = "<NUM>" if string_is_number(word) else word
            tokens.append(vocab.get(processed_word, unk_token_id))

        tokens = tokens[:max_len]
        tokens.extend([pad_token_id] * (max_len - len(tokens)))
        return tokens

    def _process_knowledge_graph(self, all_news_df: pd.DataFrame) -> None:
        """Process knowledge graph data using the KnowledgeGraphProcessor."""
        logger.info("Processing knowledge graph data...")

        kg_processor = KnowledgeGraphProcessor(
            cache_dir=self.dataset_path / "knowledge_graph",
            dataset_path=self.dataset_path,
            max_entities=self.max_entities,
            max_relations=self.max_relations,
        )

        kg_processor.process(all_news_df["title"])

        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load entity and context embeddings from files."""
        logger.info("Loading entity and context embeddings...")

        for mode in ["train", "dev", "test"]:
            entity_file = self.dataset_path / mode / "entity_embedding.vec"
            context_file = self.dataset_path / mode / "context_embedding.vec"

            if entity_file.exists():
                with open(entity_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split("\t")
                            if len(terms) == 101:
                                self.entity_embeddings[terms[0]] = list(map(float, terms[1:]))

            if context_file.exists():
                with open(context_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split("\t")
                            if len(terms) == 101:
                                self.context_embeddings[terms[0]] = list(map(float, terms[1:]))

    def generate_dataset_summary(self) -> None:
        """Generate a comprehensive dataset summary CSV file with processed data statistics."""
        logger.info("Generating dataset summary...")

        processed_dir = self.dataset_path / "processed"
        processed_dir.mkdir(exist_ok=True)

        try:
            summary_data = {}

            logger.info("Collecting basic dataset info...")
            collect_basic_dataset_info(self, summary_data)

            logger.info("Collecting news statistics...")
            collect_news_statistics(self, summary_data)

            logger.info("Collecting behavior statistics...")
            collect_behavior_statistics(self, summary_data)

            logger.info("Collecting overall statistics...")
            collect_overall_statistics(self, summary_data)

            logger.info("Collecting quality metrics...")
            collect_quality_metrics(summary_data)

            logger.info("Creating DataFrame and saving to CSV...")
            summary_df = pd.DataFrame([summary_data])
            summary_df = reorder_summary_columns(summary_df)

            summary_file_path = processed_dir / "datasets_summary.csv"
            summary_df.to_csv(summary_file_path, index=False)

            logger.info("Saving unique user IDs to CSV...")
            save_unique_users_to_csv(self)

            logger.info(f"Dataset summary saved to: {summary_file_path}")
            logger.info(f"Summary contains {len(summary_data)} statistics")

            log_key_statistics(summary_data)

            return summary_file_path

        except Exception as e:
            logger.error(f"Error generating dataset summary: {e}")
            logger.error("Continuing without summary generation...")
            return None

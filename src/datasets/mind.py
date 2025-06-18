import logging
import pickle
import urllib.request
import zipfile
import os
from typing import Dict, Tuple, List, Optional, Set, Any, Union
import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
import tensorflow as tf
from tensorflow import keras

from datasets.base import BaseNewsDataset
from utils.cache_manager import CacheManager
from utils.embeddings import EmbeddingsManager
from utils.logging import setup_logging
from utils.sampling import ImpressionSampler
from datasets.utils import display_statistics, apply_data_fraction, string_is_number
from datasets.dataloader import (
    NewsDataLoader,
    UserHistoryBatchDataloader,
    NewsBatchDataloader,
    ImpressionIterator,
)
from datasets.knowledge_graph import KnowledgeGraphProcessor

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
console = Console()


class MINDDataset(BaseNewsDataset):
    def __init__(
        self,
        name: str,
        version: str,  # "small", "large", or "200k"
        urls: Dict,
        max_title_length: int,
        max_abstract_length: int,
        max_history_length: int,
        max_impressions_length: int,
        seed: int,
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
        word_threshold: int = 3,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
        max_entities: int = 1000,
        max_relations: int = 500,
    ):
        super().__init__()
        self.name = name
        self.version = version
        self.use_knowledge_graph = use_knowledge_graph
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
        policy = keras.mixed_precision.global_policy()
        self.float_dtype = tf.dtypes.as_dtype(policy.compute_dtype)

        # Store validation split parameters
        self.validation_split_strategy = validation_split_strategy
        self.validation_split_percentage = validation_split_percentage
        self.validation_split_seed = (
            validation_split_seed if validation_split_seed is not None else seed
        )

        # Knowledge graph related attributes
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.entity_embeddings: Dict[str, list] = {}
        self.context_embeddings: Dict[str, list] = {}
        self.entity_embedding_relation: Dict[str, Set] = collections.defaultdict(set)

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

        # Set random seed
        np.random.seed(seed)

        # Load and process news data (will build vocab and tokenize)
        self.processed_news = self.process_news()

        # Load and process behaviors data
        self._load_data(mode)

    @property
    def train_size(self) -> int:
        """Number of training behaviors"""
        return (
            len(self.train_behaviors_data["impression_ids"])
            if "impression_ids" in self.train_behaviors_data
            else 0
        )

    @property
    def val_size(self) -> int:
        """Number of validation behaviors"""
        return (
            len(self.val_behaviors_data["impression_ids"])
            if "impression_ids" in self.val_behaviors_data
            else 0
        )

    @property
    def test_size(self) -> int:
        """Number of test behaviors"""
        return (
            len(self.test_behaviors_data["impression_ids"])
            if "impression_ids" in self.test_behaviors_data
            else 0
        )

    def process_news(self) -> Dict[str, np.ndarray]:
        """Process news articles into numerical format, building vocabulary and tokenizing titles.

        This method orchestrates several key steps if processed data is not found in cache:
        1.  Downloads the raw MIND dataset (if not already present).
        2.  om train, dev (and test if available for the dataset version).
        3.  Builds a vocabularyReads news data fr (`self.vocab`) based on word frequencies in *training news titles*:
            a.  Initializes vocab with `[PAD]:0` and `[UNK]:1`.
            b.  Counts word frequencies from tokenized training titles. Numbers are mapped to a `<NUM>` token.
            c.  Optionally adds `<NUM>` to the vocab if its frequency meets `self.word_threshold`.
            d.  Iterates through words sorted by frequency. Words are added to the vocabulary
                if they meet `self.word_threshold` and are not already special tokens (PAD, UNK, NUM).
                This step filters out rare words to create a more manageable and potentially more robust vocabulary.
            e.  The resulting vocabulary and its size are logged.
        4.  Tokenizes all news titles (from train, dev, test) using the built vocabulary:
            a.  Each title is converted into a sequence of integer token IDs.
            b.  OOV (Out-Of-Vocabulary) words are mapped to the `[UNK]` token's ID.
            c.  Sequences are padded/truncated to `self.max_title_length`.
        5.  Creates a filtered embedding matrix aligned with the built vocabulary:
            a.  Loads pre-trained GloVe embeddings.
            b.  Initializes PAD vector to zeros and UNK/<NUM> (if in vocab) vectors with values derived
                from GloVe statistics (mean/std for random initialization if not in GloVe).
            c.  For other vocab words, uses their GloVe vector if available, otherwise initializes randomly.
        6.  Saves the processed vocabulary, tokenized titles, and embedding matrix to cache for future runs.

        If cached data exists, it loads it directly.
        Finally, it populates `self.news_id_str_to_tokens` (mapping news ID string to tokenized title array)
        and returns a dictionary containing processed news data including tokens, embeddings, and vocab.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing processed news data,
                including 'tokens', 'embeddings', and 'vocab'.
        """
        processed_path = self.dataset_path / "processed"
        os.makedirs(processed_path, exist_ok=True)

        # Define file names based on new parameters if they affect caching
        vocab_file = processed_path / f"vocab_thresh{self.word_threshold}.pkl"
        processed_news_file = processed_path / f"processed_news_thresh{self.word_threshold}.pkl"
        embeddings_file = processed_path / f"filtered_embeddings_thresh{self.word_threshold}.npy"

        self.download_dataset()  # Ensure raw data is present

        if not (processed_news_file.exists() and embeddings_file.exists() and vocab_file.exists()):
            logger.info(f"Processing news data with word_threshold={self.word_threshold}...")

            logger.info("Reading all news (train, dev, test) for vocab building...")
            # 1. Read all news (train, dev, test) for vocab building
            news_dfs = []
            train_news_path = self.dataset_path / "train" / "news.tsv"
            dev_news_path = (
                self.dataset_path / "valid" / "news.tsv"
            )  # MIND small uses 'valid' for dev

            test_news_path = None
            # For MIND Small, dev set (valid/news.tsv) is often used for testing.
            # For MIND Large, a separate test set might exist.
            if "large" in self.version.lower():
                potential_test_path = self.dataset_path / "test" / "news.tsv"
                if potential_test_path.exists():
                    test_news_path = potential_test_path

            df_names = [
                "id",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ]

            # ---- Join all news dataframes ----
            train_news_df = pd.read_csv(
                train_news_path, sep="\\t", header=None, names=df_names, na_filter=False
            )
            news_dfs.append(train_news_df)

            dev_news_df = pd.read_csv(
                dev_news_path, sep="\\t", header=None, names=df_names, na_filter=False
            )
            news_dfs.append(dev_news_df)

            if test_news_path:
                test_news_df_content = pd.read_csv(
                    test_news_path, sep="\\t", header=None, names=df_names, na_filter=False
                )
                news_dfs.append(test_news_df_content)

            all_news_df = pd.concat(news_dfs, ignore_index=True).drop_duplicates(
                subset=["id"], keep="first"
            )
            logger.info("All news dataframes joined and duplicates dropped...")

            # Load knowledge graph if enabled
            if self.use_knowledge_graph:
                self._process_knowledge_graph(all_news_df)

            # Create a mapping from original news string ID to a continuous integer ID for array indexing
            # This is important if news IDs are not continuous or not integers
            unique_news_ids_str = all_news_df["id"].unique()
            self.news_str_id_to_int_idx = {
                nid_str: i for i, nid_str in enumerate(unique_news_ids_str)
            }
            logger.info("News string ID to integer index mapping created...")

            # Create category and subcategory mappings
            logger.info("Creating category and subcategory mappings...")
            unique_categories = sorted(all_news_df["category"].unique())
            unique_subcategories = sorted(all_news_df["subcategory"].unique())

            self.category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
            self.subcategory_to_idx = {
                subcat: idx for idx, subcat in enumerate(unique_subcategories)
            }

            logger.info(
                f"Found {len(unique_categories)} unique categories and {len(unique_subcategories)} unique subcategories"
            )

            # Create category and subcategory arrays
            category_indices = np.zeros(len(unique_news_ids_str), dtype=np.int32)
            subcategory_indices = np.zeros(len(unique_news_ids_str), dtype=np.int32)

            for nid_str, cat, subcat in zip(
                all_news_df["id"], all_news_df["category"], all_news_df["subcategory"]
            ):
                int_idx = self.news_str_id_to_int_idx[nid_str]
                category_indices[int_idx] = self.category_to_idx[cat]
                subcategory_indices[int_idx] = self.subcategory_to_idx[subcat]

            logger.info("Building vocabulary from news titles...")
            word_counter = collections.Counter()

            # Use only training news titles to build vocabulary
            logger.info(
                f"Counting words from {len(train_news_df):,} training news titles for vocabulary construction..."
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
                    "Counting words in train titles...", total=len(train_news_df)
                )
                for title_text in train_news_df["title"].values:  # Corrected to use train_news_df
                    words = self._segment_text_into_words(str(title_text))
                    for word in words:
                        processed_word = "<NUM>" if string_is_number(word) else word
                        word_counter[processed_word] += 1
                    progress.advance(task)

            logger.info(f"Found {len(word_counter):,} unique word tokens before thresholding.")
            logger.info(
                f"Top 20 most common words before thresholding: {word_counter.most_common(20)}"
            )

            self.vocab = {"[PAD]": 0, "[UNK]": 1}
            token_id_counter = 2  # Start after PAD and UNK
            logger.info(f"Initial vocab: {self.vocab}, next token ID: {token_id_counter}")

            # Add <NUM> token if it meets threshold or if we decide to always include it
            if "<NUM>" in word_counter and word_counter["<NUM>"] >= self.word_threshold:
                self.vocab["<NUM>"] = token_id_counter
                token_id_counter += 1
                logger.info(
                    f"<NUM> token added to vocab. Vocab: {self.vocab}, next token ID: {token_id_counter}"
                )
            elif "<NUM>" in word_counter:  # Log if <NUM> is present but below threshold
                logger.info(
                    f"'<NUM>' token count {word_counter['<NUM>']} is below threshold {self.word_threshold}. Not adding to vocab unless explicitly handled."
                )
            else:
                logger.info("<NUM> token not found in word counts.")

            sorted_word_counts = sorted(
                word_counter.items(), key=lambda item: item[1], reverse=True
            )
            logger.info(
                f"Starting to build final vocabulary by filtering {len(sorted_word_counts)} unique word entries with threshold {self.word_threshold}..."
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
                    "Filtering and adding words to vocab...", total=len(sorted_word_counts)
                )
                for word, count in sorted_word_counts:
                    if word in self.vocab:  # Already handled (PAD, UNK, NUM)
                        progress.advance(task)
                        continue
                    if count >= self.word_threshold:
                        self.vocab[word] = token_id_counter
                        token_id_counter += 1
                    progress.advance(
                        task
                    )  # Ensure advancement even if word is not added due to threshold

            logger.info(
                f"Vocabulary size (after threshold {self.word_threshold}): {len(self.vocab)}"
            )
            with open(vocab_file, "wb") as f:
                pickle.dump(self.vocab, f)
            logger.info(f"Vocab saved to {vocab_file}")

            logger.info("Tokenizing all news titles with the new vocabulary...")
            num_unique_news = len(unique_news_ids_str)
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
                    # Tokenize title
                    tokenized_titles_np[int_idx] = self.tokenize_text(
                        str(title_text),  # Ensure text is string
                        self.vocab,
                        self.max_title_length,
                        unk_token_id=self.vocab["[UNK]"],
                        pad_token_id=self.vocab["[PAD]"],
                    )
                    # Tokenize abstract
                    tokenized_abstracts_np[int_idx] = self.tokenize_text(
                        str(abstract_text),  # Ensure text is string
                        self.vocab,
                        self.max_abstract_length,
                        unk_token_id=self.vocab["[UNK]"],
                        pad_token_id=self.vocab["[PAD]"],
                    )
                    progress.advance(task)

            logger.info("Loading GloVe embeddings and creating filtered embedding matrix...")
            # Use the method that returns raw GloVe tensor and its vocab map
            glove_tensor_tf, glove_vocab_map = (
                self.embeddings_manager.load_glove_embeddings_tf_and_vocab_map(self.embedding_size)
            )
            if glove_tensor_tf is None or glove_vocab_map is None:
                raise ValueError(
                    "GloVe embeddings or vocab map could not be loaded by EmbeddingsManager."
                )

            logger.info("Calculating GloVe mean and std...")
            # Perform calculations on the large raw GloVe tensor on the CPU
            # to prevent potential GPU memory issues.
            with tf.device("/cpu:0"):
                glove_mean_np = tf.reduce_mean(glove_tensor_tf, axis=0).numpy()
                glove_std_np = tf.math.reduce_std(glove_tensor_tf, axis=0).numpy()
            logger.info("GloVe mean and std calculated successfully.")

            logger.info("Creating inital embedding matrix...")
            embedding_matrix = np.zeros(
                (len(self.vocab), self.embedding_size), dtype=self.float_dtype.as_numpy_dtype
            )

            # Initialize PAD vector
            embedding_matrix[self.vocab["[PAD]"]] = np.zeros(
                self.embedding_size, dtype=self.float_dtype.as_numpy_dtype
            )

            # Initialize UNK vector
            embedding_matrix[self.vocab["[UNK]"]] = np.random.normal(
                loc=glove_mean_np, scale=glove_std_np, size=self.embedding_size
            ).astype(self.float_dtype.as_numpy_dtype)

            logger.info(
                "Making sure <NUM> vector is in the embedding matrix and tf.gather runs on CPU..."
            )
            # Force tf.gather operations using the CPU-bound glove_tensor_tf to also run on CPU.
            # This avoids copying the large glove_tensor_tf to GPU for these lookup operations.
            with tf.device("/cpu:0"):
                # Initialize <NUM> vector if it's in our vocab
                if "<NUM>" in self.vocab:
                    num_token_id = self.vocab["<NUM>"]
                    glove_num_idx = glove_vocab_map.get("<NUM>")
                    if glove_num_idx is not None:
                        embedding_matrix[num_token_id] = tf.gather(
                            glove_tensor_tf, glove_num_idx
                        ).numpy()
                    else:
                        glove_number_idx = glove_vocab_map.get("number")
                        if glove_number_idx is not None:
                            embedding_matrix[num_token_id] = tf.gather(
                                glove_tensor_tf, glove_number_idx
                            ).numpy()
                        else:  # Fallback to random
                            embedding_matrix[num_token_id] = np.random.normal(
                                loc=glove_mean_np, scale=glove_std_np, size=self.embedding_size
                            ).astype(self.float_dtype.as_numpy_dtype)
                logger.info("Initial embedding matrix created successfully.")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "Populating inital embedding matrix...", total=len(self.vocab)
                    )
                    for word, idx in self.vocab.items():
                        if word in ["[PAD]", "[UNK]", "<NUM>"]:  # Already handled
                            progress.advance(task)
                            continue
                        glove_word_idx = glove_vocab_map.get(word)
                        if glove_word_idx is not None:
                            embedding_matrix[idx] = tf.gather(
                                glove_tensor_tf, glove_word_idx
                            ).numpy()
                        else:  # Word not in GloVe, initialize randomly
                            embedding_matrix[idx] = np.random.normal(
                                loc=glove_mean_np, scale=glove_std_np, size=self.embedding_size
                            ).astype(self.float_dtype.as_numpy_dtype)
                        progress.advance(task)

            np.save(embeddings_file, embedding_matrix)
            logger.info(f"Embedding matrix created successfully and saved to {embeddings_file}")

            processed_news_content: Dict[str, Any] = {
                "news_ids_original_strings": unique_news_ids_str,  # Store the original string IDs
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
            logger.info(
                f"Loading processed news data from cache (threshold {self.word_threshold})..."
            )
            with open(vocab_file, "rb") as f:
                self.vocab = pickle.load(f)
            with open(processed_news_file, "rb") as f:
                processed_news_content = pickle.load(f)

            # Rebuild self.news_str_id_to_int_idx from loaded data
            self.news_str_id_to_int_idx = {
                nid_str: i
                for i, nid_str in enumerate(processed_news_content["news_ids_original_strings"])
            }

        logger.info(f"Loading filtered embeddings (threshold {self.word_threshold})...")
        embedding_matrix_loaded = np.load(embeddings_file)
        processed_news_content["embeddings"] = tf.constant(
            embedding_matrix_loaded, dtype=self.float_dtype
        )

        # This map is used by process_behaviors. It maps original news ID string to tokenized title array
        self.news_id_str_to_tokens: Dict[str, np.ndarray] = {
            nid_str: processed_news_content["tokens"][self.news_str_id_to_int_idx[nid_str]]
            for nid_str in processed_news_content["news_ids_original_strings"]
        }
        # Also add vocab to the returned dict for the model
        processed_news_content["vocab"] = self.vocab

        return processed_news_content

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

    def _sample_users(self, sample_num: int = 200000) -> List[str]:
        """Sample users from the dataset.

        Args:
            sample_num: Number of users to sample

        Returns:
            List of sampled user IDs
        """
        logger.info(f"Sampling {sample_num} users...")

        # Collect all unique users
        user_set = set()

        # Collect users from train set
        train_behaviors = pd.read_csv(
            self.dataset_path / "train" / "behaviors.tsv",
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )
        user_set.update(train_behaviors["user_id"].unique())

        # Collect users from dev set
        dev_behaviors = pd.read_csv(
            self.dataset_path / "dev" / "behaviors.tsv",
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )
        user_set.update(dev_behaviors["user_id"].unique())

        # Convert to list and shuffle
        user_list = list(user_set)
        np.random.shuffle(user_list)

        # Sample users
        if sample_num > len(user_list):
            logger.warning(
                f"Requested sample size {sample_num} is larger than available users {len(user_list)}"
            )
            sample_num = len(user_list)

        sampled_users = user_list[:sample_num]

        # Save sampled users
        sampled_users_file = self.dataset_path / "sampled_users.json"
        with open(sampled_users_file, "w", encoding="utf-8") as f:
            json.dump(sampled_users, f)

        return sampled_users

    def _process_data(self) -> None:
        """Process train/val/test data and save to disk."""
        processed_path = self.dataset_path / "processed"

        # Sample users if using 200k version
        if self.version == "200k":
            sampled_users = self._sample_users()
            sampled_user_set = set(sampled_users)
        else:
            sampled_user_set = None

        # ----- Process train data
        logger.info("Processing train data...")
        train_behaviors_dict, val_behaviors_dict = self.get_train_val_data(sampled_user_set)
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
        test_behaviors_dict = self.get_test_data(sampled_user_set)

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

        # Get URLs from configuration based on version
        if not self.urls:
            raise ValueError(f"No URLs found in configuration for version: {self.version}")

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

        # Download knowledge graph if enabled
        if self.use_knowledge_graph:
            self._download_knowledge_graph()

        # Add to cache index
        self.cache_manager.add_to_cache(
            "mind",
            self.version,
            "dataset",
            metadata={
                "splits": list(self.urls.keys()),
                "max_title_length": self.max_title_length,
                "max_history_length": self.max_history_length,
                "version": self.version,
                "use_knowledge_graph": self.use_knowledge_graph,
            },
        )

    def _download_and_unzip_file(
        self, url: str, zip_path: Path, extract_path: Path, description: str
    ) -> None:
        """Downloads a file from a URL and unzips it."""
        logger.info(f"Downloading {description} data from {url} to {zip_path}...")
        # Create download directory if it doesn't exist
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        urllib.request.urlretrieve(url, zip_path)

        logger.info(f"Extracting {description} to {extract_path}...")
        # Extract the file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Clean up
        zip_path.unlink()
        logger.info(f"Successfully downloaded and extracted {description} data.")

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

    def process_behaviors(
        self, behaviors_df: pd.DataFrame, stage: str
    ) -> Dict[str, Union[np.ndarray, list]]:
        """Process behaviors storing only tokens, not embeddings."""
        histories_news_ids: List[list] = (
            []
        )  # List of news IDs in the history of news articles clicked by the user
        history_news_tokens: List[list] = []  # Tokens for each news article in the history
        history_news_abstract_tokens: List[
            list
        ] = []  # Abstract tokens for each news article in the history
        history_news_categories: List[list] = []  # Categories for each news article in the history
        history_news_subcategories: List[
            list
        ] = []  # Subcategories for each news article in the history
        candidate_news_ids: List[list] = []  # Candidate news IDs for each impression row
        candidate_news_tokens: List[list] = []  # Tokens for each candidate news article
        candidate_news_abstract_tokens: List[
            list
        ] = []  # Abstract tokens for each candidate news article
        candidate_news_categories: List[list] = []  # Categories for each candidate news article
        candidate_news_subcategories: List[
            list
        ] = []  # Subcategories for each candidate news article
        labels: List[list] = []  # Labels for each candidate news article
        impression_ids: List[int] = []  # IDs for each impression. That's the row in the behaviors.tsv file
        user_ids: List[str] = []  # IDs for each user

        # Statistics tracking
        total_original_rows = len(behaviors_df)
        total_positives = 0
        rows_with_multiple_positives = 0
        max_positives_in_row = 0

        # Create a lookup dictionary from news ID to pre-tokenized titles
        news_tokens: Dict[int, np.ndarray] = dict(
            zip(
                [
                    int(nid.split("N")[1])
                    for nid in self.processed_news["news_ids_original_strings"]
                ],
                self.processed_news["tokens"],
            )
        )

        # Create a lookup dictionary from news ID to pre-tokenized abstracts
        news_abstract_tokens: Dict[int, np.ndarray] = dict(
            zip(
                [
                    int(nid.split("N")[1])
                    for nid in self.processed_news["news_ids_original_strings"]
                ],
                self.processed_news["abstract_tokens"],
            )
        )

        # Create lookup dictionaries for categories and subcategories
        news_categories: Dict[int, int] = dict(
            zip(
                [
                    int(nid.split("N")[1])
                    for nid in self.processed_news["news_ids_original_strings"]
                ],
                self.processed_news["category_indices"],
            )
        )
        news_subcategories: Dict[int, int] = dict(
            zip(
                [
                    int(nid.split("N")[1])
                    for nid in self.processed_news["news_ids_original_strings"]
                ],
                self.processed_news["subcategory_indices"],
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

            # Parse all behavior rows
            for _, row in behaviors_df.iterrows():
                # Count positives in this row
                impressions = str(row["impressions"]).split()
                user_id = str(row["user_id"]).split("U")[1]
                positives_count = sum(imp.split("-")[1] == "1" for imp in impressions)

                total_positives += positives_count
                if positives_count > 1:
                    rows_with_multiple_positives += 1
                max_positives_in_row = max(max_positives_in_row, positives_count)

                # -- Process history
                history = str(row["history"]).split() if pd.notna(row["history"]) else []
                history = history[-self.max_history_length :]

                # -- Store indices/tokens instead of embeddings
                history_nid_list = [
                    int(h.split("N")[1]) for h in history
                ]  # List of news IDs in the history clicked news clicked by the user
                curr_history_tokens = [news_tokens[h_idx] for h_idx in history_nid_list]
                curr_history_abstract_tokens = [
                    news_abstract_tokens[h_idx] for h_idx in history_nid_list
                ]
                curr_history_categories = [news_categories[h_idx] for h_idx in history_nid_list]
                curr_history_subcategories = [
                    news_subcategories[h_idx] for h_idx in history_nid_list
                ]

                # -- Pad histories
                history_pad_length = self.max_history_length - len(history)
                history_nid_list = [0] * history_pad_length + history_nid_list
                curr_history_tokens = [
                    [0] * self.max_title_length
                ] * history_pad_length + curr_history_tokens
                curr_history_abstract_tokens = [
                    [0] * self.max_abstract_length
                ] * history_pad_length + curr_history_abstract_tokens
                curr_history_categories = [0] * history_pad_length + curr_history_categories
                curr_history_subcategories = [0] * history_pad_length + curr_history_subcategories

                # -- Process candidate news
                # cand_nid_group_list: List of candidate news groups, where each group contains:
                #   - 1 positive news article
                #   - k negative news articles (k = max_impressions_length - 1)
                #   - Articles are shuffled within each group
                # label_group_list: List of label groups corresponding to cand_nid_group_list, where:
                #   - 1 represents a positive article (click)
                #   - 0 represents a negative article (non-click)
                cand_nid_group_list, label_group_list = self.sampler.sample_candidates_news(
                    stage=stage,
                    candidates=impressions,
                    random_train_samples=self.random_train_samples,
                )

                if stage == "train":
                    # -- For each group (1 positive : k negatives)
                    # Some rows have multiple positive articles, so we need to repeat the same history for each group
                    for cand_nid_group, label_group in zip(cand_nid_group_list, label_group_list):
                        # Repeat the same history for this group
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
                    # -- For validation/testing, keep original impressions without padding
                    cand_nid_group, label_group = cand_nid_group_list, label_group_list

                    # Store original impressions without padding
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

            # Convert to numpy arrays for training data, keep as lists for val/test
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
                    "labels": np.array(labels, dtype=self.float_dtype.as_numpy_dtype),
                    "impression_ids": np.array(impression_ids, dtype=np.int32),
                    "user_ids": np.array(user_ids, dtype=np.int32),
                }
            else:
                # For val/test, keep as lists to handle variable lengths
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

            # -- Calculate statistics
            total_processed_rows = len(histories_news_ids)
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
            logger.info(f"Total users: {len(set(user_ids))}")

            return result

    def get_news(self) -> pd.DataFrame:
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
        sampled_user_set: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, Union[np.ndarray, list]], Dict[str, Union[np.ndarray, list]]]:
        """Load and process training data, splitting into train and validation sets."""
        # Process behaviors
        behaviors_df = pd.read_csv(
            self.dataset_path / "train" / "behaviors.tsv",
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        # Filter by sampled users if provided
        if sampled_user_set is not None:
            behaviors_df = behaviors_df[behaviors_df["user_id"].isin(list(sampled_user_set))]

        if self.validation_split_strategy == "random":
            logger.info(
                f"Using random split for validation: {self.validation_split_percentage*100}% of training behaviors data, seed: {self.validation_split_seed}"
            )
            # Shuffle the DataFrame
            shuffled_df = behaviors_df.sample(
                frac=1, random_state=self.validation_split_seed
            ).reset_index(drop=True)

            # Split into new train and validation sets
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
            # Convert time to datetime
            behaviors_df["time"] = pd.to_datetime(behaviors_df["time"])
            # Split into train and validation based on the last day
            last_day = behaviors_df["time"].max().date()
            train_behaviors = behaviors_df[behaviors_df["time"].dt.date < last_day]
            val_behaviors = behaviors_df[behaviors_df["time"].dt.date == last_day]
        else:
            raise ValueError(f"Unknown validation_split_strategy: {self.validation_split_strategy}")

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

    def get_test_data(
        self,
        sampled_user_set: Optional[Set[str]] = None,
    ) -> Dict[str, Union[np.ndarray, list]]:
        """Load and process test data"""
        test_dir = self.dataset_path / "valid"  # Use valid folder for testing

        test_behaviors = pd.read_csv(
            test_dir / "behaviors.tsv",
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        # Filter by sampled users if provided
        if sampled_user_set is not None:
            test_behaviors = test_behaviors[
                test_behaviors["user_id"].isin(list(sampled_user_set))
            ]

        logger.info(f"Test behaviors: {len(test_behaviors):,}")

        # Process test behaviors
        return self.process_behaviors(
            test_behaviors,
            stage="test",
        )

    def train_dataloader(self, batch_size: int) -> tf.data.Dataset:
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
            batch_size=batch_size,
            process_title=self.process_title,
            process_abstract=self.process_abstract,
            process_category=self.process_category,
            process_subcategory=self.process_subcategory,
        )

    def user_history_dataloader(self, mode: str) -> UserHistoryBatchDataloader:
        """Create dataloader for user history validation/testing."""
        if mode == "val":
            user_history_tokens = self.val_behaviors_data["history_news_tokens"]
            user_history_abstract_tokens = self.val_behaviors_data["history_news_abstract_tokens"]
            user_history_category = self.val_behaviors_data["history_news_categories"]
            user_history_subcategory = self.val_behaviors_data["history_news_subcategories"]
            impression_ids = self.val_behaviors_data["impression_ids"]
        elif mode == "test":
            user_history_tokens = self.test_behaviors_data["history_news_tokens"]
            user_history_abstract_tokens = self.test_behaviors_data["history_news_abstract_tokens"]
            user_history_category = self.test_behaviors_data["history_news_categories"]
            user_history_subcategory = self.test_behaviors_data["history_news_subcategories"]
            impression_ids = self.test_behaviors_data["impression_ids"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return UserHistoryBatchDataloader(
            history_tokens=user_history_tokens,
            history_abstract_tokens=user_history_abstract_tokens,
            history_category=user_history_category,
            history_subcategory=user_history_subcategory,
            impression_ids=impression_ids,
            batch_size=512,
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

    def news_dataloader(self) -> NewsBatchDataloader:
        """Create dataloader for processed news validation/testing."""
        news_ids = self.processed_news.get("news_ids_original_strings", np.array([]))
        news_tokens = self.processed_news.get("tokens", np.array([]))
        news_abstract_tokens = self.processed_news.get("abstract_tokens", np.array([]))
        news_category_indices = self.processed_news.get("category_indices", np.array([]))
        news_subcategory_indices = self.processed_news.get("subcategory_indices", np.array([]))

        return NewsBatchDataloader(
            news_ids=tf.constant(news_ids),
            news_tokens=tf.constant(news_tokens),
            news_abstract_tokens=tf.constant(news_abstract_tokens),
            news_category_indices=tf.constant(news_category_indices),
            news_subcategory_indices=tf.constant(news_subcategory_indices),
            batch_size=512,
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
        """Segment a sentence string into a list of word strings.

        Args:
            sent: The sentence to be segmented.

        Returns:
            List of word strings from the sentence.
        """
        # treat consecutive words or special punctuation as words
        pat = re.compile(r"[\w]+|[.,!?;|]")
        return pat.findall(sent.lower()) if isinstance(sent, str) else []

    def tokenize_text(
        self, text: str, vocab: Dict[str, int], max_len: int, unk_token_id: int, pad_token_id: int
    ) -> List[int]:
        """
        Converts a raw text string into a fixed-length sequence of numerical token IDs.

        This method performs several steps:
        1. Uses `self._segment_text_into_words()` to split the input `text` into a list of word strings.
        2. For each word, it checks if it represents a number (using `string_is_number`) and maps it
            to a special "<NUM>" token if so.
        3. Looks up each processed word string in the provided `vocab` to get its integer ID.
        4. Maps any out-of-vocabulary (OOV) words to the `unk_token_id`.
        5. Truncates or pads the resulting list of token IDs with `pad_token_id` to ensure
            it has length `max_len`.

        Args:
            text: The raw input string to tokenize.
            vocab: A dictionary mapping word strings to their integer token IDs.
            max_len: The desired fixed length for the output token ID sequence.
            unk_token_id: The integer ID to use for OOV words (Out-Of-Vocabulary words).
            pad_token_id: The integer ID to use for padding.

        Returns:
            A list of integer token IDs of length `max_len`.
        """
        words = self._segment_text_into_words(text)
        tokens = []
        for word in words:
            processed_word = "<NUM>" if string_is_number(word) else word
            tokens.append(vocab.get(processed_word, unk_token_id))

        tokens = tokens[:max_len]
        tokens.extend([pad_token_id] * (max_len - len(tokens)))
        return tokens

    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Build vocabulary from texts."""
        vocab = {"[PAD]": 0}
        token_id = 1

        for text in texts:
            words = self._segment_text_into_words(text)
            for word in words:
                if word not in vocab:
                    vocab[word] = token_id
                    token_id += 1

        return vocab

    def _process_knowledge_graph(self, all_news_df: pd.DataFrame) -> None:
        """Process knowledge graph data using the KnowledgeGraphProcessor."""
        logger.info("Processing knowledge graph data...")

        # Initialize knowledge graph processor
        kg_processor = KnowledgeGraphProcessor(
            cache_dir=self.dataset_path / "knowledge_graph",
            dataset_path=self.dataset_path,
            max_entities=self.max_entities,
            max_relations=self.max_relations,
        )

        # Process knowledge graph
        kg_processor.process(all_news_df["title"])

        # Load processed embeddings
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load entity and context embeddings from files."""
        logger.info("Loading entity and context embeddings...")

        # Load entity embeddings
        for mode in ["train", "dev", "test"]:
            entity_file = self.dataset_path / mode / "entity_embedding.vec"
            context_file = self.dataset_path / mode / "context_embedding.vec"

            if entity_file.exists():
                with open(entity_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split("\t")
                            if len(terms) == 101:  # entity + 100-dimensional embedding
                                self.entity_embeddings[terms[0]] = list(map(float, terms[1:]))

            if context_file.exists():
                with open(context_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split("\t")
                            if len(terms) == 101:  # entity + 100-dimensional embedding
                                self.context_embeddings[terms[0]] = list(map(float, terms[1:]))

import logging
from typing import Dict, Tuple, Optional

import numpy as np
import requests
import tensorflow as tf
from rich.progress import Progress
import zipfile
import urllib3
from tensorflow import keras

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from utils.cache_manager import CacheManager

logger = logging.getLogger("embeddings")


class EmbeddingsManager:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.glove_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.embedding_matrix: Optional[tf.Tensor] = None
        self.vocab_size: Optional[int] = None
        self.embedding_dim: Optional[int] = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.category_embeddings: Optional[tf.Variable] = None
        self.subcategory_embeddings: Optional[tf.Variable] = None
        policy = keras.mixed_precision.global_policy()
        self.float_dtype = tf.dtypes.as_dtype(policy.compute_dtype)

    def load_glove(self, dim: int = 300) -> None:
        """Load GloVe embeddings and create embedding matrix"""
        path = self.cache_manager.get_embedding_path("glove", dim)
        txt_file = path / f"glove.840B.{dim}d.txt"
        npy_file = path / f"glove.840B.{dim}d.npy"

        # If embeddings already loaded, return early
        if self.embedding_matrix is not None:
            return

        # Try to load from .npy if it exists
        if npy_file.exists():
            logger.info("Loading GloVe embeddings from .npy file...")
            self.glove_embeddings = np.load(npy_file, allow_pickle=True).item()
            if self.glove_embeddings:
                logger.info(f"Loaded {len(self.glove_embeddings):,} word vectors from .npy")

            # Create embedding matrix
            self._create_embedding_matrix(dim)
            return

        # Check if txt file exists, if not download
        if not txt_file.exists():
            self._download_and_extract_glove_zip(path, dim, txt_file)

        # Load embeddings from txt file
        logger.info("Loading GloVe embeddings from txt file...")
        self.glove_embeddings = {}

        with Progress() as progress:
            file_size = txt_file.stat().st_size
            task = progress.add_task("Loading embeddings...", total=file_size)

            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        values = line.split()
                        # Join all elements except the last 300 as the word
                        word = "".join(values[:-dim])
                        # Take last 300 elements as the embedding
                        vector = np.asarray(values[-dim:], dtype=self.float_dtype.as_numpy_dtype)
                        self.glove_embeddings[word] = vector
                        progress.advance(task, len(line.encode("utf-8")))
                    except Exception as e:
                        logger.warning(f"Error processing line: {line[:50]}... Error: {str(e)}")
                        continue

        # Save embeddings for faster future loading
        logger.info("Saving embeddings to .npy format...")
        if self.glove_embeddings is not None:
            np.save(npy_file, self.glove_embeddings)  # type: ignore

        # Create embedding matrix
        self._create_embedding_matrix(dim)

    def _download_and_extract_glove_zip(self, path, dim, txt_file) -> None:
        """Downloads and extracts GloVe embeddings if the txt file doesn't exist."""
        # Download if not present
        url = f"https://nlp.stanford.edu/data/glove.840B.{dim}d.zip"
        zip_path = path / f"glove.840B.{dim}d.zip"
        path.mkdir(parents=True, exist_ok=True)

        # Download the zip file
        with Progress() as progress:
            task = progress.add_task("Downloading GloVe...", total=None)
            # Disable SSL verification to handle expired certificates
            response = requests.get(url, stream=True, verify=False)
            total_size = int(response.headers.get("content-length", 0))
            progress.update(task, total=total_size)

            with open(zip_path, "wb") as f:
                for data in response.iter_content(chunk_size=4096):
                    f.write(data)
                    progress.advance(task, len(data))

        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path)
            logger.info(f"Extracted GloVe embeddings to {path}")

        # Clean up zip file
        zip_path.unlink()

    def _create_embedding_matrix(self, dim: int) -> None:
        """Create embedding matrix from loaded GloVe embeddings.

        This matrix can be used as a pre-trained weight matrix for an embedding layer.
        The 0-th row is reserved for padding. For other rows, the index `i`
        corresponds to the GloVe vector of the word that was assigned index `i`
        during the enumeration of `self.glove_embeddings.items()`.
        The resulting `self.embedding_matrix` is a TensorFlow constant.

        Args:
            dim: The dimension of the GloVe embeddings.
        """
        if self.glove_embeddings is None:
            return

        self.embedding_dim = dim
        self.vocab_size = len(self.glove_embeddings)

        # Create embedding matrix
        embedding_matrix_np = np.zeros(
            (self.vocab_size + 1, dim), dtype=self.float_dtype.as_numpy_dtype
        )  # +1 for padding
        for idx, (word, vector) in enumerate(self.glove_embeddings.items(), 1):
            embedding_matrix_np[idx] = vector

        # Convert to TensorFlow constant
        self.embedding_matrix = tf.constant(embedding_matrix_np, dtype=self.float_dtype)  # type: ignore

    def get_glove_raw_data(
        self, dim: int = 300
    ) -> Tuple[Optional[tf.Tensor], Optional[Dict[str, int]]]:
        """
        Ensures GloVe embeddings are loaded and returns the raw GloVe vectors as a TF tensor
        and a word-to-index map for that tensor.
        The tensor contains all words from the loaded GloVe file.

        Args:
            dim: The dimension of the GloVe embeddings.

        Returns:
            A tuple containing:
            - A TF tensor of shape (vocab_size, dim) containing the GloVe embeddings.
            - A dictionary mapping words to their indices in the embedding tensor.
        """
        if self.glove_embeddings is None:
            self.load_glove(dim)

        if self.glove_embeddings is None:  # Still None after trying to load
            logger.error("GloVe embeddings could not be loaded.")
            return None, None

        logger.info("Creating GloVe tensor and word-to-index map...")
        glove_words = list(self.glove_embeddings.keys())
        glove_vectors_list = [self.glove_embeddings[word] for word in glove_words]

        if not glove_vectors_list:
            logger.error("No GloVe vectors found in self.glove_embeddings.")
            return None, None

        # Force the creation of this very large tensor on the CPU.
        # This is a one-time setup step to get all raw GloVe vectors.
        # Placing it on CPU prevents potential GPU OOM errors when TF tries to allocate it.
        try:
            with tf.device("/cpu:0"):
                raw_glove_tensor = tf.constant(glove_vectors_list, dtype=self.float_dtype)
        except Exception as e:
            logger.error(f"Error converting GloVe vectors list to tensor: {e}")
            # Fallback: try creating from individual tensors if there's a shape mismatch issue
            try:
                with tf.device("/cpu:0"):
                    raw_glove_tensor = tf.stack(
                        [tf.constant(v, dtype=self.float_dtype) for v in glove_vectors_list]
                    )
            except Exception as e_stack:
                logger.error(f"Critical error: Could not create raw_glove_tensor: {e_stack}")
                return None, None

        word_to_idx_map = {word: i for i, word in enumerate(glove_words)}
        logger.info("GloVe tensor and word-to-index map created successfully.")

        return raw_glove_tensor, word_to_idx_map

    # Alias the new method name to what process_news expects
    def load_glove_embeddings_tf_and_vocab_map(
        self, dim: int = 300
    ) -> Tuple[Optional[tf.Tensor], Optional[Dict[str, int]]]:
        """
        Loads GloVe embeddings and returns them as a TF tensor and a word-to-index map.

        Args:
            dim: The dimension of the GloVe embeddings.

        Returns:
            A tuple containing:
            - A TF tensor of shape (vocab_size, dim) containing the GloVe embeddings.
            - A dictionary mapping words to their indices in the embedding tensor.
        """
        return self.get_glove_raw_data(dim)

    def create_category_embeddings(
        self, num_categories: int, embedding_dim: int = 100
    ) -> tf.Variable:
        """Create trainable category embeddings.
        
        Args:
            num_categories: Number of unique categories
            embedding_dim: Dimension of category embeddings
            
        Returns:
            TensorFlow tensor of shape (num_categories, embedding_dim)
        """
        # Initialize with random normal distribution
        initializer = tf.keras.initializers.GlorotNormal()
        self.category_embeddings = tf.Variable(
            initializer(shape=(num_categories, embedding_dim)),
            trainable=True,
            name="category_embeddings",
            dtype=self.float_dtype,
        )
        return self.category_embeddings

    def create_subcategory_embeddings(
        self, num_subcategories: int, embedding_dim: int = 100
    ) -> tf.Variable:
        """Create trainable subcategory embeddings.
        
        Args:
            num_subcategories: Number of unique subcategories
            embedding_dim: Dimension of subcategory embeddings
            
        Returns:
            TensorFlow tensor of shape (num_subcategories, embedding_dim)
        """
        # Initialize with random normal distribution
        initializer = tf.keras.initializers.GlorotNormal()
        self.subcategory_embeddings = tf.Variable(
            initializer(shape=(num_subcategories, embedding_dim)),
            trainable=True,
            name="subcategory_embeddings",
            dtype=self.float_dtype,
        )
        return self.subcategory_embeddings

    def get_category_embeddings(self) -> Optional[tf.Variable]:
        """Get category embeddings if they exist."""
        return self.category_embeddings

    def get_subcategory_embeddings(self) -> Optional[tf.Variable]:
        """Get subcategory embeddings if they exist."""
        return self.subcategory_embeddings

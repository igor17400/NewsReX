import logging
from typing import Dict, Tuple, Optional
import tarfile

import numpy as np
import requests
import keras
from rich.progress import Progress
import zipfile
import urllib3

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from .cache_manager import CacheManager

logger = logging.getLogger("embeddings")


class EmbeddingsManager:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.glove_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.embedding_matrix: Optional[keras.KerasTensor] = None
        self.vocab_size: Optional[int] = None
        self.embedding_dim: Optional[int] = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.category_embeddings: Optional[keras.Variable] = None
        self.subcategory_embeddings: Optional[keras.Variable] = None
        policy = keras.mixed_precision.global_policy()
        # Use compute dtype from mixed precision policy  
        if policy.compute_dtype == "mixed_float16":
            self.float_dtype = "float16"
        elif policy.compute_dtype == "float16":
            self.float_dtype = "float16"
        else:
            self.float_dtype = "float32"

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
            logger.info(f"Loading GloVe embeddings from .npy file: {npy_file}")
            try:
                loaded_data = np.load(npy_file, allow_pickle=True)
                if hasattr(loaded_data, 'item'):
                    self.glove_embeddings = loaded_data.item()
                else:
                    self.glove_embeddings = loaded_data
                    
                if self.glove_embeddings and isinstance(self.glove_embeddings, dict):
                    logger.info(f"Loaded {len(self.glove_embeddings):,} word vectors from .npy")
                    # Create embedding matrix
                    self._create_embedding_matrix(dim)
                    return
                else:
                    logger.error(f"Loaded data is not a valid dictionary. Type: {type(self.glove_embeddings)}")
                    self.glove_embeddings = None
            except Exception as e:
                logger.error(f"Failed to load .npy file: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.glove_embeddings = None

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
                        # Handle the case where the word might contain spaces
                        # Join all elements except the last `dim` as the word
                        word = " ".join(values[:-dim])
                        # Take last `dim` elements as the embedding
                        vector = np.asarray(values[-dim:], dtype=self.float_dtype)
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
            (self.vocab_size + 1, dim), dtype=self.float_dtype
        )  # +1 for padding
        for idx, (word, vector) in enumerate(self.glove_embeddings.items(), 1):
            embedding_matrix_np[idx] = vector

        # Convert to TensorFlow constant
        self.embedding_matrix = keras.ops.convert_to_tensor(embedding_matrix_np)  # type: ignore

    def get_glove_raw_data(
        self, dim: int = 300
    ) -> Tuple[Optional[keras.KerasTensor], Optional[Dict[str, int]]]:
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
            logger.info(f"GloVe embeddings not loaded yet, loading with dim={dim}...")
            self.load_glove(dim)

        if self.glove_embeddings is None:  # Still None after trying to load
            logger.error("GloVe embeddings could not be loaded after calling load_glove().")
            return None, None

        logger.info(f"Creating GloVe tensor and word-to-index map from {len(self.glove_embeddings)} words...")
        try:
            glove_words = list(self.glove_embeddings.keys())
            glove_vectors_list = [self.glove_embeddings[word] for word in glove_words]
        except Exception as e:
            logger.error(f"Error extracting words/vectors from glove_embeddings: {e}")
            return None, None

        if not glove_vectors_list:
            logger.error("No GloVe vectors found in self.glove_embeddings.")
            return None, None

        # Force the creation of this very large tensor on the CPU.
        # This is a one-time setup step to get all raw GloVe vectors.
        # Placing it on CPU prevents potential GPU OOM errors when JAX tries to allocate it.
        logger.info(f"Converting {len(glove_vectors_list)} GloVe vectors to tensor...")
        try:
            # Convert to numpy first to ensure CPU placement, then to tensor
            glove_array = np.array(glove_vectors_list, dtype=np.float32)
            logger.info(f"Created numpy array with shape {glove_array.shape}")
            raw_glove_tensor = keras.ops.convert_to_tensor(glove_array)
            logger.info(f"Successfully created tensor with shape {raw_glove_tensor.shape}")
        except Exception as e:
            logger.warning(f"Error converting GloVe vectors list to tensor: {e}")
            # Fallback: try creating tensor in smaller chunks to manage memory
            logger.info("Trying chunked approach to handle memory constraints...")
            try:
                chunk_size = 10000  # Process in chunks of 10k embeddings
                tensor_chunks = []
                num_chunks = (len(glove_vectors_list) + chunk_size - 1) // chunk_size
                
                for i in range(0, len(glove_vectors_list), chunk_size):
                    chunk_idx = i // chunk_size + 1
                    logger.info(f"Processing chunk {chunk_idx}/{num_chunks}...")
                    chunk = glove_vectors_list[i:i+chunk_size]
                    chunk_array = np.array(chunk, dtype=np.float32)
                    tensor_chunks.append(keras.ops.convert_to_tensor(chunk_array))
                
                logger.info("Concatenating chunks...")
                raw_glove_tensor = keras.ops.concatenate(tensor_chunks, axis=0)
                logger.info(f"Successfully created tensor with shape {raw_glove_tensor.shape}")
            except Exception as e_stack:
                logger.error(f"Critical error: Could not create raw_glove_tensor: {e_stack}")
                import traceback
                logger.error(traceback.format_exc())
                return None, None

        word_to_idx_map = {word: i for i, word in enumerate(glove_words)}
        logger.info("GloVe tensor and word-to-index map created successfully.")

        return raw_glove_tensor, word_to_idx_map

    # Alias the new method name to what process_news expects
    def load_glove_embeddings_tf_and_vocab_map(
        self, dim: int = 300
    ) -> Tuple[Optional[keras.KerasTensor], Optional[Dict[str, int]]]:
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
    ):
        """Create trainable category embeddings.
        
        Args:
            num_categories: Number of unique categories
            embedding_dim: Dimension of category embeddings
            
        Returns:
            TensorFlow tensor of shape (num_categories, embedding_dim)
        """
        # Initialize with random normal distribution
        initializer = keras.initializers.GlorotNormal()
        self.category_embeddings = keras.Variable(
            initializer(shape=(num_categories, embedding_dim)),
            trainable=True,
            name="category_embeddings",
            dtype=self.float_dtype,
        )
        return self.category_embeddings

    def create_subcategory_embeddings(
        self, num_subcategories: int, embedding_dim: int = 100
    ):
        """Create trainable subcategory embeddings.
        
        Args:
            num_subcategories: Number of unique subcategories
            embedding_dim: Dimension of subcategory embeddings
            
        Returns:
            TensorFlow tensor of shape (num_subcategories, embedding_dim)
        """
        # Initialize with random normal distribution
        initializer = keras.initializers.GlorotNormal()
        self.subcategory_embeddings = keras.Variable(
            initializer(shape=(num_subcategories, embedding_dim)),
            trainable=True,
            name="subcategory_embeddings",
            dtype=self.float_dtype,
        )
        return self.subcategory_embeddings

    def get_category_embeddings(self) -> Optional[keras.Variable]:
        """Get category embeddings if they exist."""
        return self.category_embeddings

    def get_subcategory_embeddings(self) -> Optional[keras.Variable]:
        """Get subcategory embeddings if they exist."""
        return self.subcategory_embeddings

    def load_bpemb(self, language: str, vocab_size: int = 200000, dim: int = 300) -> Dict[str, np.ndarray]:
        """Load BPEmb embeddings from pre-trained files.
        
        Args:
            language: Language code (e.g., 'ja', 'de', 'fr')
            vocab_size: Vocabulary size (default: 200000)
            dim: Embedding dimension (default: 300)
            
        Returns:
            Dictionary mapping BPE tokens to embedding vectors
        """
        path = self.cache_manager.get_embedding_path("bpemb", f"{language}_{vocab_size}_{dim}")
        txt_file = path / f"{language}.wiki.bpe.vs{vocab_size}.d{dim}.w2v.txt"
        npy_file = path / f"{language}.wiki.bpe.vs{vocab_size}.d{dim}.w2v.npy"
        
        # Try to load from .npy if it exists
        if npy_file.exists():
            logger.info(f"Loading BPEmb embeddings from .npy file: {npy_file}")
            try:
                bpemb_embeddings = np.load(npy_file, allow_pickle=True).item()
                if bpemb_embeddings and isinstance(bpemb_embeddings, dict):
                    logger.info(f"Loaded {len(bpemb_embeddings):,} BPE tokens from .npy")
                    return bpemb_embeddings
                else:
                    logger.error(f"Loaded data is not a valid dictionary. Type: {type(bpemb_embeddings)}")
            except Exception as e:
                logger.error(f"Failed to load .npy file: {e}")
        
        # Check if txt file exists, if not download
        if not txt_file.exists():
            self._download_and_extract_bpemb(path, language, vocab_size, dim)
        
        # Load embeddings from txt file
        logger.info(f"Loading BPEmb embeddings from txt file: {txt_file}")
        bpemb_embeddings = {}
        
        with Progress() as progress:
            file_size = txt_file.stat().st_size
            task = progress.add_task("Loading BPEmb embeddings...", total=file_size)
            
            with open(txt_file, "r", encoding="utf-8") as f:
                # Skip header line if present
                first_line = f.readline()
                if not first_line.split()[0].replace('-', '').isdigit():
                    # First line contains vocab size and dimension, skip it
                    pass
                else:
                    # First line is an embedding, process it
                    f.seek(0)
                
                for line in f:
                    try:
                        values = line.strip().split()
                        if len(values) <= dim:
                            continue
                        
                        # First element is the BPE token, rest are embeddings
                        token = values[0]
                        vector = np.asarray(values[1:dim+1], dtype=self.float_dtype)
                        bpemb_embeddings[token] = vector
                        progress.advance(task, len(line.encode("utf-8")))
                    except Exception as e:
                        logger.warning(f"Error processing line: {line[:50]}... Error: {str(e)}")
                        continue
        
        # Save embeddings for faster future loading
        logger.info("Saving BPEmb embeddings to .npy format...")
        if bpemb_embeddings:
            np.save(npy_file, bpemb_embeddings)
        
        logger.info(f"Loaded {len(bpemb_embeddings):,} BPE token embeddings")
        return bpemb_embeddings
    
    def _download_and_extract_bpemb(self, path, language: str, vocab_size: int, dim: int) -> None:
        """Download and extract BPEmb embeddings."""
        # BPEmb download URL pattern
        url = f"https://bpemb.h-its.org/{language}/{language}.wiki.bpe.vs{vocab_size}.d{dim}.w2v.txt.tar.gz"
        tar_path = path / f"{language}.wiki.bpe.vs{vocab_size}.d{dim}.w2v.txt.tar.gz"
        path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading BPEmb embeddings for {language} from {url}")
        
        # Download the tar.gz file
        with Progress() as progress:
            task = progress.add_task(f"Downloading BPEmb {language}...", total=None)
            response = requests.get(url, stream=True, verify=False)
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to download BPEmb embeddings. Status code: {response.status_code}")
            
            total_size = int(response.headers.get("content-length", 0))
            progress.update(task, total=total_size)
            
            with open(tar_path, "wb") as f:
                for data in response.iter_content(chunk_size=4096):
                    f.write(data)
                    progress.advance(task, len(data))
        
        # Extract the tar.gz file
        logger.info(f"Extracting BPEmb embeddings to {path}")
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(path)
        
        # Move the extracted file to the expected location
        # The tar contains nested directories like data/ja/ja.wiki.bpe.vs200000.d300.w2v.txt
        extracted_file = path / "data" / language / f"{language}.wiki.bpe.vs{vocab_size}.d{dim}.w2v.txt"
        target_file = path / f"{language}.wiki.bpe.vs{vocab_size}.d{dim}.w2v.txt"
        
        if extracted_file.exists():
            import shutil
            shutil.move(str(extracted_file), str(target_file))
            # Clean up the empty data directory
            data_dir = path / "data"
            if data_dir.exists():
                shutil.rmtree(data_dir)
        
        # Clean up tar.gz file
        tar_path.unlink()
        logger.info(f"Successfully downloaded and extracted BPEmb embeddings for {language}")
    
    def get_bpemb_embeddings(self, language: str, vocab_size: int = 200000, dim: int = 300) -> Dict[str, np.ndarray]:
        """Get BPEmb embeddings, loading them if necessary.
        
        Args:
            language: Language code (e.g., 'ja', 'de', 'fr')
            vocab_size: Vocabulary size (default: 200000)
            dim: Embedding dimension (default: 300)
            
        Returns:
            Dictionary mapping BPE tokens to embedding vectors
        """
        return self.load_bpemb(language, vocab_size, dim)

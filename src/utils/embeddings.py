import logging
from typing import Dict, Tuple

import numpy as np
import requests
import tensorflow as tf
from rich.progress import Progress
from transformers import BertTokenizer, TFBertModel
import zipfile

from utils.cache_manager import CacheManager

logger = logging.getLogger("embeddings")


class EmbeddingsManager:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.glove_embeddings = None
        self.embedding_matrix = None
        self.vocab_size = None
        self.embedding_dim = None
        self.bert_model = None
        self.bert_tokenizer = None

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
            logger.info(f"Loaded {len(self.glove_embeddings):,} word vectors from .npy")

            # Create embedding matrix
            self._create_embedding_matrix(dim)
            return

        # Check if txt file exists, if not download
        if not txt_file.exists():
            # Download if not present
            url = f"https://nlp.stanford.edu/data/glove.840B.{dim}d.zip"
            zip_path = path / f"glove.840B.{dim}d.zip"
            path.mkdir(parents=True, exist_ok=True)

            # Download the zip file
            with Progress() as progress:
                task = progress.add_task("Downloading GloVe...", total=None)
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                progress.update(task, total=total_size)

                with open(zip_path, "wb") as f:
                    for data in response.iter_content(chunk_size=4096):
                        f.write(data)
                        progress.advance(task, len(data))

            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(path)
                logger.info(f"Extracted GloVe embeddings to {path}")
            
            # Clean up zip file
            zip_path.unlink()

        # Load embeddings from txt file
        logger.info("Loading GloVe embeddings from txt file...")
        self.glove_embeddings = {}
        
        with Progress() as progress:
            file_size = txt_file.stat().st_size
            task = progress.add_task("Loading embeddings...", total=file_size)
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        values = line.split()
                        # Join all elements except the last 300 as the word
                        word = ''.join(values[:-dim])
                        # Take last 300 elements as the embedding
                        vector = np.asarray(values[-dim:], dtype='float32')
                        self.glove_embeddings[word] = vector
                        progress.advance(task, len(line.encode('utf-8')))
                    except Exception as e:
                        logger.warning(f"Error processing line: {line[:50]}... Error: {str(e)}")
                        continue

        # Save embeddings for faster future loading
        logger.info("Saving embeddings to .npy format...")
        np.save(npy_file, self.glove_embeddings)

        # Create embedding matrix
        self._create_embedding_matrix(dim)

    def _create_embedding_matrix(self, dim: int) -> None:
        """Create embedding matrix from loaded GloVe embeddings"""
        self.embedding_dim = dim
        self.vocab_size = len(self.glove_embeddings)
        
        # Create embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size + 1, dim))  # +1 for padding
        for idx, (word, vector) in enumerate(self.glove_embeddings.items(), 1):
            self.embedding_matrix[idx] = vector
            
        # Convert to TensorFlow constant
        self.embedding_matrix = tf.constant(self.embedding_matrix, dtype=tf.float32)

    def create_filtered_embedding_matrix(self, vocab: Dict[str, int], dim: int = 300) -> np.ndarray:
        """Create embedding matrix only for words in the provided vocabulary"""
        logger.info("Creating filtered embedding matrix for vocabulary...")
        
        # Initialize matrix with random values for unknown words
        matrix_size = len(vocab)
        embedding_matrix = np.random.uniform(-0.25, 0.25, (matrix_size, dim))
        
        # Set zero vector for padding token
        embedding_matrix[0] = np.zeros(dim)
        
        # Count how many words we found
        found_words = 0
        
        for word, idx in vocab.items():
            if word in self.glove_embeddings:
                embedding_matrix[idx] = self.glove_embeddings[word]
                found_words += 1
        
        logger.info(f"Found embeddings for {found_words}/{len(vocab)} words")
        return tf.constant(embedding_matrix, dtype=tf.float32)

    def load_bert(self, model_name: str = "bert-base-uncased") -> None:
        """Load BERT model and tokenizer"""
        if self.bert_model is None:
            logger.info(f"Loading BERT model: {model_name}")
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert_model = TFBertModel.from_pretrained(model_name)

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
        self.bert_model = None
        self.bert_tokenizer = None

    def load_glove(self, dim: int = 300) -> Dict[str, np.ndarray]:
        """Load GloVe embeddings from disk or download if not present"""
        path = self.cache_manager.get_embedding_path("glove", dim)
        txt_file = path / f"glove.840B.{dim}d.txt"
        npy_file = path / f"glove.840B.{dim}d.npy"

        # If embeddings already loaded, return them
        if self.glove_embeddings is not None:
            return self.glove_embeddings

        # Try to load from .npy if it exists
        if npy_file.exists():
            logger.info("Loading GloVe embeddings from .npy file...")
            embeddings = {}
            vectors = np.load(npy_file, allow_pickle=True).item()  # Load as dictionary
            logger.info(f"Loaded {len(vectors):,} word vectors from .npy")
            self.glove_embeddings = vectors
            return vectors

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
        embeddings = {}
        
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
                        embeddings[word] = vector
                        progress.advance(task, len(line.encode('utf-8')))
                    except Exception as e:
                        logger.warning(f"Error processing line: {line[:50]}... Error: {str(e)}")
                        continue

        # Save as .npy for faster future loading
        logger.info("Saving embeddings to .npy format...")
        np.save(npy_file, embeddings)  # Save the entire dictionary
        
        logger.info(f"Loaded {len(embeddings):,} word vectors")
        self.glove_embeddings = embeddings
        return embeddings

    def load_bert(self, model_name: str = "bert-base-uncased") -> Tuple[TFBertModel, BertTokenizer]:
        """Load BERT model and tokenizer"""
        if self.bert_model is None:
            logger.info(f"Loading BERT model: {model_name}")
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert_model = TFBertModel.from_pretrained(model_name)
        return self.bert_model, self.bert_tokenizer

    def get_bert_embeddings(self, texts: list, max_length: int = 512) -> tf.Tensor:
        """Get BERT embeddings for texts"""
        model, tokenizer = self.load_bert()

        # Tokenize texts
        encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="tf")

        # Get embeddings
        outputs = model(encoded)
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

import logging
from typing import Dict, Tuple

import numpy as np
import requests
import tensorflow as tf
from rich.progress import Progress
from transformers import BertTokenizer, TFBertModel

from utils.cache_manager import CacheManager

logger = logging.getLogger("embeddings")


class EmbeddingsManager:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.glove_embeddings = None
        self.bert_model = None
        self.bert_tokenizer = None

    def load_glove(self, dim: int = 300) -> Dict[str, np.ndarray]:
        """Load GloVe embeddings"""
        if self.cache_manager.is_embedding_cached("glove", dim):
            return np.load(self.cache_manager.get_embedding_path("glove", dim) / "embeddings.npy")

        logger.info(f"Downloading GloVe embeddings ({dim}d)...")
        url = f"https://nlp.stanford.edu/data/glove.6B.{dim}d.txt"

        embeddings_dict = {}
        with Progress() as progress:
            task = progress.add_task("Downloading GloVe...", total=None)

            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            progress.update(task, total=total_size)

            path = self.cache_manager.get_embedding_path("glove", dim)
            path.mkdir(parents=True, exist_ok=True)

            with open(path / "glove.txt", "wb") as f:
                for data in response.iter_content(chunk_size=4096):
                    f.write(data)
                    progress.update(task, advance=len(data))

            # Load embeddings
            with open(path / "glove.txt", "r", encoding="utf-8") as f:
                for line in progress.track(f, description="Loading embeddings..."):
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype="float32")
                    embeddings_dict[word] = vector

        # Save to cache
        np.save(path / "embeddings.npy", embeddings_dict)
        self.cache_manager.add_to_cache("glove", str(dim), "embedding")

        return embeddings_dict

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
        encoded = tokenizer(
            texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="tf"
        )

        # Get embeddings
        outputs = model(encoded)
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

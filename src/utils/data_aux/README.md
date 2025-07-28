# Data Utilities

This module contains utilities for data processing, embeddings management, sampling strategies, and caching.

## Files

### 🔤 `embeddings.py`
Manages word embeddings (GloVe, BERT) for the news recommendation models.

**Classes:**
- `EmbeddingsManager` - Handles loading and managing word embeddings

**Features:**
- GloVe embeddings support (50d, 100d, 200d, 300d)
- BERT tokenizer integration
- Category/subcategory embeddings
- Automatic downloading and caching
- Mixed precision support

### 🎲 `sampling.py`
Impression sampling strategies for training data.

**Classes:**
- `ImpressionSampler` - Base class for sampling strategies
- Various sampling strategies (random, popularity-based, etc.)

**Features:**
- Configurable sampling strategies
- Support for different sampling distributions
- Reproducible sampling with seed control

### 💾 `cache_manager.py`
Efficient caching system for datasets and embeddings.

**Classes:**
- `CacheManager` - Manages cached data and embeddings

**Features:**
- Automatic cache directory management
- Cache invalidation strategies
- Compressed storage support
- Fast loading of preprocessed data

## Usage Examples

```python
# Embeddings management
from src.utils.data.embeddings import EmbeddingsManager

embeddings_mgr = EmbeddingsManager(cache_manager)
embeddings_mgr.load_glove(dim=300)
embedding_matrix = embeddings_mgr.build_embedding_matrix(word2idx)

# Sampling
from src.utils.data.sampling import ImpressionSampler

sampler = ImpressionSampler(config)
sampled_impressions = sampler.sample(impressions, n_samples=5)

# Cache management
from src.utils.data.cache_manager import CacheManager

cache_mgr = CacheManager(cache_dir=".cache")
cached_data = cache_mgr.load_or_process("news_data", process_fn)
```

## Data Flow

1. **Raw Data** → Cache Manager → Processed Data
2. **Embeddings** → Cache → Model Initialization
3. **Impressions** → Sampler → Training Batches

## Configuration

- `cache.root_dir` - Root directory for cache storage
- `cache.clear_on_start` - Whether to clear cache on startup
- `dataset.embedding_type` - Type of embeddings to use (glove/bert)
- `dataset.embedding_size` - Dimension of embeddings
# Utils Module

This module contains utility functions and classes organized by their functionality to support the BTC (Behind The Curtains) news recommendation framework.

## Directory Structure

### 📊 `metrics/`
News recommendation specific metrics (AUC, MRR, nDCG) and Keras metric wrappers.
- `functions_optimized.py` - Core metric computation functions
- `wrapper.py` - Keras 3 compatible metric wrappers

### 🚀 `training/`
Training loop orchestration, callbacks, and training utilities.
- `orchestration.py` - Main training loop orchestrator using Keras model.fit()
- `callbacks.py` - Custom Keras callbacks for evaluation
- `engine.py` - Training and test step functions
- `losses.py` - Custom loss functions

### 📈 `evaluation/`
Model evaluation utilities and validation functions.
- `evaluation.py` - Evaluation epoch runners and validation functions

### 💾 `data/`
Data processing, embeddings, and caching utilities.
- `embeddings.py` - GloVe/BERT embeddings management
- `sampling.py` - Impression sampling strategies
- `cache_manager.py` - Caching system for datasets and embeddings

### 🖥️ `device/`
Device configuration and setup for different backends.
- `device.py` - JAX/GPU device configuration
- `device_configs/` - (Reserved for device-specific configurations)

### 📝 `io/`
Input/output utilities for saving models, predictions, and logging.
- `saving.py` - Model and prediction saving utilities
- `logging.py` - Logging configuration and utilities

### 🔧 `model/`
Model initialization and management utilities.
- `model.py` - Model and dataset initialization

## Usage

Import utilities from their respective subdirectories:

```python
# Metrics
from src.utils.metrics.functions_optimized import NewsRecommenderMetricsOptimized as NewsRecommenderMetrics
from src.utils.metrics.wrapper import create_news_metrics

# Training
from src.utils.training.orchestration import training_loop_orchestrator
from src.utils.training.callbacks import FastEvaluationCallback

# Evaluation
from src.utils.evaluation.main import run_evaluation_epoch

# Data
from src.utils.data.embeddings import EmbeddingsManager
from src.utils.data.cache_manager import CacheManager

# Device
from src.utils.device.device import setup_device

# I/O
from src.utils.io.saving import save_run_summary_fn
from src.utils.io.logging import setup_logging
```

## Design Principles

1. **Modularity**: Each subdirectory contains related functionality
2. **Clear Naming**: Descriptive names that indicate purpose
3. **Single Responsibility**: Each module has a focused purpose
4. **Easy Discovery**: Organized structure makes finding utilities intuitive
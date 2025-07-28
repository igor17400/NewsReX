"""
NewsRex - News Recommendation Framework Utilities

This module provides organized utilities for the news recommendation system,
structured by functionality for better maintainability and discoverability.

Submodules:
    - metrics: News recommendation metrics (AUC, MRR, nDCG) and Keras wrappers
    - training: Training orchestration, callbacks, and training steps
    - evaluation: Model evaluation and validation utilities
    - data: Data processing, embeddings, and caching utilities
    - device: Device configuration for JAX backend
    - io: Logging and saving utilities
    - model: Model initialization utilities
"""

# Import commonly used functions at the package level for convenience
from .metrics.functions import NewsRecommenderMetrics
from .metrics.wrapper import create_news_metrics, LightweightNewsMetrics

from .training.orchestration import training_loop_orchestrator
from .training.callbacks import FastEvaluationCallback, SlowEvaluationCallback

from .evaluation.main import run_evaluation_epoch, get_main_comparison_metric

from .data.embeddings import EmbeddingsManager
from .data.cache_manager import CacheManager

from .device.device import setup_device

from .io.logging import setup_logging, setup_wandb_session
from .io.saving import save_run_summary_fn, get_output_run_dir

from .model.model import initialize_model_and_dataset

__all__ = [
    # Metrics
    "NewsRecommenderMetrics",
    "create_news_metrics", 
    "LightweightNewsMetrics",
    
    # Training
    "training_loop_orchestrator",
    "FastEvaluationCallback",
    "SlowEvaluationCallback",
    
    # Evaluation
    "run_evaluation_epoch",
    "get_main_comparison_metric",
    
    # Data
    "EmbeddingsManager",
    "CacheManager",
    
    # Device
    "setup_device",
    
    # I/O
    "setup_logging",
    "setup_wandb_session", 
    "save_run_summary_fn",
    "get_output_run_dir",
    
    # Model
    "initialize_model_and_dataset"
]
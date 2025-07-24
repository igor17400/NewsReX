# Evaluation Utilities

This module contains utilities for model evaluation, validation, and testing.

## Files

### 📊 `evaluation.py`
Core evaluation functions for running validation and testing.

**Key Functions:**
- `run_evaluation_epoch()` - Runs evaluation for one epoch (slow evaluation)
- `_run_initial_validation()` - Performs initial validation before training
- `_run_epoch_evaluation()` - Runs evaluation after each training epoch
- `_run_final_testing()` - Performs final testing with best model
- `get_main_comparison_metric()` - Calculates average metric for model comparison

## Features

### Fast vs Slow Evaluation
- **Fast Evaluation**: Uses precomputed news and user vectors for efficient evaluation
- **Slow Evaluation**: Traditional batch-wise evaluation without precomputation

### Evaluation Modes
- **Initial Validation**: Baseline metrics before training starts
- **Epoch Validation**: After each training epoch
- **Final Testing**: On test set with best model weights

### Metrics Computed
- Loss
- AUC (Area Under Curve)
- MRR (Mean Reciprocal Rank)
- nDCG@5 and nDCG@10 (Normalized Discounted Cumulative Gain)

## Usage Examples

```python
# Run evaluation epoch
from src.utils.evaluation.evaluation import run_evaluation_epoch

metrics = run_evaluation_epoch(
    model=model,
    eval_dataloader=val_dataloader,
    custom_metrics_calculator=metrics_engine,
    num_total_impressions=val_size,
    progress=progress_bar,
    mode="val"
)

# Run initial validation
from src.utils.evaluation.evaluation import _run_initial_validation

initial_metrics = _run_initial_validation(
    model, dataset_provider, metrics_engine, progress_bar, cfg
)
```

## Configuration

Evaluation behavior is controlled by config parameters:
- `cfg.eval.fast_evaluation` - Use fast evaluation with precomputed vectors
- `cfg.eval.run_initial_validation` - Run validation before training
- `cfg.eval.save_predictions` - Save prediction outputs
- `cfg.eval.batch_size` - Batch size for evaluation
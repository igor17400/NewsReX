# I/O Utilities

This module contains utilities for input/output operations including logging, saving models, and storing results.

## Files

### 📝 `logging.py`
Logging configuration and utilities with Rich console support.

**Functions:**
- `setup_logging()` - Configures logging with Rich handler
- `setup_wandb_session()` - Initializes Weights & Biases logging
- `log_metrics_to_console_fn()` - Pretty prints metrics to console
- `log_epoch_summary_fn()` - Logs comprehensive epoch summary
- `log_metrics_to_wandb_fn()` - Logs metrics to WandB

**Features:**
- Rich console formatting with colors and styles
- Backend-specific logger configuration (JAX, Keras, TensorFlow)
- WandB integration for experiment tracking
- Structured metric logging

### 💾 `saving.py`
Model and results saving utilities.

**Functions:**
- `save_run_summary_fn()` - Saves complete run summary with metrics
- `save_predictions_to_file_fn()` - Saves model predictions
- `get_output_run_dir()` - Gets output directory for current run

**Features:**
- Hierarchical output directory structure
- JSON/pickle format support
- Prediction saving for analysis
- Automatic directory creation
- Timestamped outputs

## Usage Examples

```python
# Logging setup
from src.utils.io.logging import setup_logging, setup_wandb_session

setup_logging(level="INFO")
setup_wandb_session(cfg)

# Save results
from src.utils.io.saving import save_run_summary_fn, save_predictions_to_file_fn

save_run_summary_fn(
    output_dir, cfg, initial_metrics, best_metrics, test_metrics, wandb_history
)

save_predictions_to_file_fn(
    predictions_dict, save_dir, epoch=5, mode="val"
)
```

## Output Directory Structure

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── models/
        │   ├── model_best.weights.h5
        │   └── model_last.weights.h5
        ├── predictions/
        │   ├── initial_val/
        │   ├── val/
        │   └── test/
        └── run_summary.json
```

## Logging Configuration

### Console Logging
- Uses Rich for beautiful console output
- Color-coded log levels
- Progress bars and spinners
- Structured metric display

### Backend Logger Levels
- `keras`: INFO - Shows model compilation and training info
- `jax`: WARNING - Reduces compilation noise
- `jaxlib`: WARNING - Reduces low-level JAX messages
- `hydra`: WARNING - Reduces configuration messages

### WandB Integration
- Automatic metric tracking
- Hyperparameter logging
- Training curves visualization
- Model performance comparison

## Configuration

From config:
- `logging.level` - Global logging level
- `logging.enable_wandb` - Enable WandB tracking
- `logging.project_name` - WandB project name
- `logging.experiment_name` - Experiment identifier
- `eval.save_predictions` - Save prediction outputs
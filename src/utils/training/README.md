# Training Utilities

This module contains utilities for model training, including orchestration, callbacks, and training step functions.

## Files

### 📋 `orchestration.py`
Main training loop orchestrator that manages the entire training process using Keras 3's `model.fit()` approach.

**Key Functions:**
- `training_loop_orchestrator()` - Main training loop with callbacks and evaluation
- `_setup_training_directories()` - Creates model and prediction directories

### 🔄 `callbacks.py`
Custom Keras callbacks for evaluation during training.

**Classes:**
- `FastEvaluationCallback` - Performs fast evaluation using precomputed vectors
- `SlowEvaluationCallback` - Traditional batch-wise evaluation
- `RichProgressCallback` - Integration with Rich progress bars

### ⚙️ `engine.py`
Core training and test step functions.

**Functions:**
- `train_step_fn()` - Executes a single training step
- `test_step_fn()` - Executes a single test/validation step

### 📉 `losses.py`
Custom loss functions for news recommendation.

**Functions:**
- Custom loss implementations for recommendation tasks

## Usage Examples

```python
# Training orchestration
from src.utils.training.orchestration import training_loop_orchestrator

# Custom callbacks
from src.utils.training.callbacks import FastEvaluationCallback, SlowEvaluationCallback

# Training steps
from src.utils.training.engine import train_step_fn, test_step_fn
```

## Design Notes

- Supports both fast (precomputed vectors) and slow (batch-wise) evaluation
- Integrated with Rich for beautiful progress tracking
- Compatible with Keras 3 + JAX backend
- Supports early stopping and model checkpointing
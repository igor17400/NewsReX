# Model Utilities

This module contains utilities for model initialization and management.

## Files

### 🏗️ `model.py`
Model and dataset initialization utilities.

**Functions:**
- `initialize_model_and_dataset()` - Creates model and dataset instances from config

**Features:**
- Hydra-based dynamic model instantiation
- Dataset loading and preprocessing
- Configuration validation
- Automatic model architecture selection
- Dataset-model compatibility checks

## Supported Models

The initialization system supports:
- **NRMS** - Neural News Recommendation with Multi-Head Self-Attention
- **NAML** - Neural News Recommendation with Attentive Multi-View Learning
- **LSTUR** - Long- and Short-term User Representations

## Usage Examples

```python
from src.utils.model.model import initialize_model_and_dataset

# Initialize from Hydra config
model, dataset_provider = initialize_model_and_dataset(cfg)

# Model is ready to compile and train
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## Model Initialization Flow

1. **Config Loading** → Hydra configuration specifies model class
2. **Dataset Loading** → Dataset provider initialized with config
3. **News Processing** → Dataset processes news articles
4. **Model Creation** → Model instantiated with processed news data
5. **Return** → Ready-to-use model and dataset provider

## Configuration

Model configuration in `configs/model/`:
```yaml
_target_: src.models.nrms.NRMS
embedding_size: 300
multiheads: 20
head_dim: 32
attention_hidden_dim: 200
dropout_rate: 0.2
```

Dataset configuration in `configs/dataset/`:
```yaml
_target_: src.datasets.mind.MINDDataset
version: small
embedding_type: glove
max_title_length: 32
```

## Model Requirements

Each model expects:
- Processed news data dictionary
- Embedding configuration
- Architecture-specific parameters
- Maximum sequence lengths

## Error Handling

The initialization handles:
- Missing configuration parameters
- Incompatible model-dataset combinations
- Missing embeddings or data files
- Invalid model architectures
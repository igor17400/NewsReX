# BTC: Behind The Curtains for Recommender Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

BTC is a modular and extensible framework for news recommendation systems research, implementing state-of-the-art models with a focus on reproducibility and ease of use. This project draws inspiration from the work done by [newsreclib](https://github.com/andreeaiana/newsreclib) with PytorchLightning, but we have chosen to proceed with Keras due to its widespread adoption and the fact that many state-of-the-art models are directly implemented using Keras.

## 🌟 Features

- 📚 Multiple SOTA news recommendation models
- 🔄 Easy-to-use training and evaluation pipeline
- 📦 Comprehensive metrics and evaluation
- 🎛️ Hydra-based configuration system
- 🚀 Weights & Biases integration for experiment tracking
- 🔌 Modular design for easy extension

## 🏗️ Supported Models

- **NRMS**: Neural News Recommendation with Multi-Head Self-Attention
- **NAML**: Neural News Recommendation with Attentive Multi-View Learning
- *(More models coming soon)*

## 📦 Supported Datasets

- **MIND**: Microsoft News Dataset (small and large versions)
- *(More datasets coming soon)*

## 🚀 Quick Start

### Prerequisites

1. Install Conda (if not already installed):
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

2. Create and activate a Conda environment with Python 3.11:
```bash
# Create new environment
conda create -n btc_env python=3.11
# Activate the environment
conda activate btc_env
```

3. Install Poetry (Python package manager):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

4. Verify Poetry installation:
```bash
poetry --version
```

5. Make sure to use one Python version 

```
Python >=3.11,<=3.12
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/igor17400/BTC.git
cd BTC
```

2. Configure Poetry to create virtual environment in project directory:
```bash
poetry config virtualenvs.in-project true
```

3. Install dependencies and create virtual environment:
```bash
# Create virtual environment and install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

4. Set up pre-commit hooks:
```bash
poetry run pre-commit install
```

5. You might need to install tensorflow with the following command to make it sure that it'll work with the GPUs

```bash
poetry add 'tensorflow[and-cuda]'
```

To test it out if it worked we recommend executing the following commands:

```bash
python test_tensorflow_gpu.py
```

Expected output:

```bash
✅ If TensorFlow detects a GPU, it will list it.
❌ If the output is an empty list ([]), TensorFlow is not using a GPU.
```

Note: You can also run commands without activating the shell using `poetry run`, for example:
```bash
poetry run python src/train.py
```

### Training a Model

```bash
# Train with default configuration (NRMS on MIND-small)
poetry run python src/train.py

# Train NRMS on MIND-small
poetry run python src/train.py experiment=nrms_mind_small
```

### Evaluation

```bash
# Evaluate the best model
poetry run python src/test.py experiment=nrms_mind_small
```

## 📁 Project Structure

```
BTC/
├── configs/                 # Hydra configuration files
│   ├── config.yaml         # Base configuration
│   ├── model/              # Model-specific configs
│   └── dataset/            # Dataset-specific configs
├── src/
│   ├── models/             # Model implementations
│   │   ├── base.py        # Abstract base classes
│   │   ├── nrms.py        # NRMS implementation
│   │   └── naml.py        # NAML implementation
│   ├── datasets/           # Dataset implementations
│   │   ├── base.py        # Abstract dataset class
│   │   └── mind.py        # MIND dataset
│   ├── utils/              # Utility functions
│   │   └── metrics.py     # Evaluation metrics
│   ├── train.py           # Training script
│   └── test.py            # Testing script
├── tests/                  # Unit tests
├── pyproject.toml         # Poetry configuration
└── README.md              # This file
```

## 📦 Metrics

The framework provides comprehensive evaluation metrics:
- AUC (Area Under ROC Curve)
- MRR (Mean Reciprocal Rank)
- nDCG@5 and nDCG@10 (Normalized Discounted Cumulative Gain)

## 🔧 Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Base configuration
- `configs/model/*.yaml`: Model-specific configurations
- `configs/dataset/*.yaml`: Dataset-specific configurations

Example configuration override:
```bash
poetry run python src/train.py \
    model=naml \
    dataset.dataset.version=large \
    train.batch_size=64 \
    train.learning_rate=0.001
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=src
```

## 🚀 Experiment Tracking

The framework integrates with Weights & Biases for experiment tracking:

1. Set up your W&B account
2. Enable tracking in config:
```yaml
logging:
  enable_wandb: true
  project_name: "your-project"
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📚 References

- NRMS: [Neural News Recommendation with Multi-Head Self-Attention](https://aclanthology.org/D19-1671/)
- NAML: [Neural News Recommendation with Attentive Multi-View Learning](https://www.ijcai.org/proceedings/2019/536)
- MIND: [MIND: A Large-scale Dataset for News Recommendation](https://aclanthology.org/2020.acl-main.331/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚀 Analytics and Visualization

The framework provides rich analytics and visualization capabilities:

### User Analytics
- User reading patterns and preferences
- Category and subcategory affinity
- Temporal interaction patterns
- Topic interest word clouds
- Interactive user journey timelines

### Content Analytics
- Long-tail distribution analysis
- Category and subcategory distributions
- Click-through rate analysis
- Time-of-day content preferences

### Recommendation Analytics
- Recommendation diversity metrics
- Temporal recommendation distribution
- Popularity vs. novelty analysis
- Topic diversity visualization

To generate visualizations:

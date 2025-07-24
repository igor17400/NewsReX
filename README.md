# BTC: Behind The Curtains for Recommender Systems

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

BTC is a modular and extensible framework for news recommendation systems research, implementing state-of-the-art models with a focus on reproducibility and ease of use. The framework has been optimized with **Keras 3 + JAX backend** for enhanced performance through JIT compilation and XLA acceleration. This project draws inspiration from the work done by [newsreclib](https://github.com/andreeaiana/newsreclib) with PyTorch Lightning, but we have chosen to proceed with Keras due to its widespread adoption and the fact that many state-of-the-art models are directly implemented using Keras.

## 🌟 Features

- 📚 Multiple SOTA news recommendation models
- ⚡ **JAX Backend Optimization**: Keras 3 + JAX for JIT compilation and XLA acceleration
- 🔄 Easy-to-use training and evaluation pipeline
- 📦 Comprehensive JAX-optimized metrics and evaluation
- 🎛️ Hydra-based configuration system
- 🚀 Weights & Biases integration for experiment tracking
- 🔌 Modular design for easy extension
- 🚀 **Performance**: Faster training and inference through JAX optimizations

## 🏗️ Supported Models

- **NRMS**: Neural News Recommendation with Multi-Head Self-Attention (Keras 3 + JAX)
- **NAML**: Neural News Recommendation with Attentive Multi-View Learning
- **LSTUR**: Long- and Short-term User Representations
- *(More models coming soon)*

## 📦 Supported Datasets

- **MIND**: Microsoft News Dataset (small and large versions)
- *(More datasets coming soon)*

## 🚀 Quick Start

### Prerequisites

1.  **Install Conda** (if not already installed):

    ```bash
    # Download and install Miniconda (or Anaconda)
    wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda # Install silently to your home directory
    eval "$($HOME/miniconda/bin/conda shell hook)" # Initialize conda in your shell
    ```
    *Note: Adjust the Miniconda path if you install it elsewhere.*

2.  **Create and activate a Conda environment** with Python 3.11:

    ```bash
    conda create -n btc_env python=3.11 -y
    conda activate btc_env
    ```

3.  **Install Poetry** (Python package manager) within your `btc_env`:

    ```bash
    conda install poetry -y
    ```

4.  **Verify Poetry's setup:**

    ```bash
    poetry --version
    # This should show your Poetry version (e.g., Poetry (version 1.7.1))
    ```
    *Note: We will configure Poetry to use your conda environment in the next steps.*

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/igor17400/BTC.git](https://github.com/igor17400/BTC.git)
    cd BTC
    ```

2.  **Point Poetry to your `btc_env` Conda environment:**
    This step ensures Poetry uses your existing Python 3.11 environment.

    ```bash
    # First, ensure Poetry's cache directories exist (resolves common errors)
    mkdir -p ~/.cache/pypoetry/virtualenvs

    # Now, tell Poetry to use the Python from your active conda environment
    poetry env use $(which python)
    ```
    *Expected output for `poetry env info` after this step:*
    ```
    Virtualenv
    Python:          3.11.13
    Implementation: CPython
    Path:           /root/miniconda3/envs/btc_env  # Or your specific conda env path
    Executable:     /root/miniconda3/envs/btc_env/bin/python
    Valid:          True
    ```

3.  **Install project dependencies:**
    Poetry will now install the project's dependencies into your `btc_env`.

    ```bash
    poetry install 
    ```

4.  **Activate Poetry's shell (optional but recommended for development):**
    This puts your current shell into the Poetry-managed environment.

    ```bash
    poetry shell
    ```

5.  **Set up pre-commit hooks:**

    ```bash
    pre-commit install
    ```

6.  **JAX Backend Dependencies:**
    Test if JAX is properly using your GPU:

    ```bash
    python test_jax_gpu.py
    ```

    *Expected output (example for GPU detected):*
    ```
    gpu
    ```
    *If the output is `cpu`, JAX is not using a GPU.*

*Note: You can always run commands without activating the Poetry shell using `poetry run`, for example: `poetry run python src/train.py`.*

### Training a Model

```bash
# Train with JAX backend
poetry run python src/train.py experiment=nrms_mind_small
```
import keras
from typing import Dict, Tuple

from rich.console import Console

console = Console()


def test_step_fn(
        model: keras.Model, data: Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor]
) -> Dict[str, keras.KerasTensor]:
    """Custom test step logic."""
    features, labels = data

    # Use Keras 3's backend-agnostic test step
    batch_metrics = model.test_on_batch(features, labels, return_dict=True)
    return batch_metrics

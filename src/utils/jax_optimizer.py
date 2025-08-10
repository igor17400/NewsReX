"""JAX optimization utilities to improve training performance."""

import jax
import jax.numpy as jnp
import logging

logger = logging.getLogger(__name__)


def warmup_jit_compilation(model, sample_batch):
    """Warm up JIT compilation by running a forward pass with sample data.
    
    Args:
        model: The Keras model to warm up
        sample_batch: A sample batch of data matching the model's expected input format
                     Can be either a dict of inputs or a tuple of (inputs, labels)
    """
    logger.info("Warming up JIT compilation...")

    try:
        # Handle different input formats
        if isinstance(sample_batch, tuple):
            # If it's a tuple, assume it's (inputs, labels) from dataloader
            inputs = sample_batch[0]
            labels = sample_batch[1] if len(sample_batch) > 1 else None
        else:
            # If it's already a dict, use it directly
            inputs = sample_batch
            labels = None

        # Run a forward pass to trigger compilation
        _ = model(inputs, training=False)
        _ = model(inputs, training=True)

        # If the model has internal models, warm them up too
        if hasattr(model, 'training_model') and model.training_model is not None:
            # For training model, we need to prepare the proper input format

            # Our base keys
            base_hist_key = "hist_tokens"
            base_cand_key = "cand_tokens"

            # A list of optional key pairs to check for.
            optional_key_pairs = [
                ("hist_abstract_tokens", "cand_abstract_tokens"),
                ("hist_category", "cand_category"),
                ("hist_subcategory", "cand_subcategory"),
            ]

            # Check if the base tokens exist. If not, we can't proceed.
            if base_hist_key not in inputs or base_cand_key not in inputs:
                print("Warning: Base tokens are missing. Cannot perform concatenation.")
                return None, None

            # Start with the base tokens as the initial arrays to concatenate.
            history_to_concat = [inputs[base_hist_key]]
            candidate_to_concat = [inputs[base_cand_key]]

            # Iterate through the optional keys and add them if they exist.
            for hist_key, cand_key in optional_key_pairs:
                if hist_key in inputs and cand_key in inputs:
                    # Check if we need to expand dimensions for scalar values.
                    if inputs[hist_key].ndim == 1:
                        history_to_concat.append(jnp.expand_dims(inputs[hist_key], axis=-1))
                        candidate_to_concat.append(jnp.expand_dims(inputs[cand_key], axis=-1))
                    else:
                        history_to_concat.append(inputs[hist_key])
                        candidate_to_concat.append(inputs[cand_key])

            # Perform the concatenation.
            history_concat = jnp.concatenate(history_to_concat, axis=-1)
            candidate_concat = jnp.concatenate(candidate_to_concat, axis=-1)

            # Warm up training model with concatenated inputs
            _ = model.training_model([history_concat, candidate_concat], training=False)
            _ = model.training_model([history_concat, candidate_concat], training=True)

        if hasattr(model, 'scorer_model') and model.scorer_model is not None:
            # Scorer model warmup - it has different input shapes
            # We'll skip this for now as it's less critical
            pass

        # Block until computations are complete
        jax.block_until_ready(jax.device_put(0))

        logger.info("JIT compilation warmup completed")

    except Exception as e:
        logger.warning(f"JIT warmup encountered an error (non-critical): {e}")
        # Continue anyway - warmup is optimization, not required

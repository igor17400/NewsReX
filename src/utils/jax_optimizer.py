"""JAX optimization utilities to improve training performance."""

import os
import jax
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
            if 'hist_tokens' in inputs and 'cand_tokens' in inputs:
                # Concatenate history inputs
                history_concat = jax.numpy.concatenate([
                    inputs["hist_tokens"],
                    inputs["hist_abstract_tokens"],
                    jax.numpy.expand_dims(inputs["hist_category"], axis=-1),
                    jax.numpy.expand_dims(inputs["hist_subcategory"], axis=-1),
                ], axis=-1)

                # Concatenate candidate inputs
                candidate_concat = jax.numpy.concatenate([
                    inputs["cand_tokens"],
                    inputs["cand_abstract_tokens"],
                    jax.numpy.expand_dims(inputs["cand_category"], axis=-1),
                    jax.numpy.expand_dims(inputs["cand_subcategory"], axis=-1),
                ], axis=-1)

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

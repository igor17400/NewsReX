import logging
from typing import List

import tensorflow as tf

logger = logging.getLogger(__name__)


def setup_device(
    gpu_ids: List[int], memory_limit: float = 0.9, mixed_precision: bool = True
) -> None:
    """Configure TensorFlow device and memory settings.

    Args:
        gpu_ids: List of GPU IDs to use. Empty list for CPU.
        memory_limit: Fraction of GPU memory to allocate (0 to 1)
        mixed_precision: Whether to use mixed precision training
    """
    # Disable eager execution (can help with some CUDA issues)
    # tf.compat.v1.disable_eager_execution()

    logger.info("🚀 Setting up devices...")

    if not gpu_ids:
        logger.info("💻 Using CPU for training")
        tf.config.set_visible_devices([], "GPU")
        return

    try:
        # Get available GPUs
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            logger.warning("⚠️ No GPU found. Using CPU instead.")
            return

        # Set visible devices
        visible_gpus = [gpus[i] for i in gpu_ids if i < len(gpus)]
        tf.config.set_visible_devices(visible_gpus, "GPU")

        logger.info(f"Found {len(gpus)} GPUs, using {len(visible_gpus)}")

        # Configure memory growth and limits
        for i, gpu in enumerate(visible_gpus):
            try:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)

                # Set memory limit if requested
                if memory_limit < 1.0:
                    memory_in_mb = int(40 * 1024 * memory_limit)  # Assuming 40GB A100
                    tf.config.set_logical_device_configuration(
                        gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_in_mb)]
                    )

                logger.info(f"🎮 GPU {i}: {gpu.name}")
                logger.info(f"   - Memory limit: {memory_limit*100:.0f}%")

            except RuntimeError as e:
                logger.error(f"❌ Error configuring GPU {i}: {e}")
                continue

        # Enable mixed precision if requested
        if mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("🚀 Enabled mixed precision training")
            logger.info(f"Compute dtype: {policy.compute_dtype}")
            logger.info(f"Variable dtype: {policy.variable_dtype}")

    except RuntimeError as e:
        logger.error(f"❌ Error setting up GPU: {e}")
        logger.info("⚠️ Falling back to CPU")
        tf.config.set_visible_devices([], "GPU")

    # Final device check
    logical_devices = tf.config.list_logical_devices()
    for device in logical_devices:
        logger.info(f"🔍 Available device: {device.name}")

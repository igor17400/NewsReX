import logging
from typing import List

import tensorflow as tf

logger = logging.getLogger(__name__)


def setup_device(gpu_ids: List[int], memory_limit: float = 0.9, mixed_precision: bool = True) -> None:
    """Configure TensorFlow device and memory settings.

    Args:
        gpu_ids: List of GPU IDs to use. Empty list for CPU.
        memory_limit: Fraction of GPU memory to allocate (0 to 1)
        mixed_precision: Whether to use mixed precision training
    """
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

        # Configure memory growth
        for gpu in visible_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            if memory_limit < 1.0:
                # Set memory limit using virtual device configuration
                memory_in_mb = int(24 * 1024 * memory_limit)  # Assuming 24GB GPU, adjust if needed
                tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_in_mb)])

        # Enable mixed precision if requested
        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("🚀 Enabled mixed precision training")

        logger.info(f"🎮 Using GPU(s): {gpu_ids}")
        for i, gpu in enumerate(visible_gpus):
            logger.info(f"📊 GPU {i}: {gpu.name}, Memory limit: {memory_limit*100:.0f}%")

    except RuntimeError as e:
        logger.error(f"❌ Error setting up GPU: {e}")
        logger.info("⚠️ Falling back to CPU")
        tf.config.set_visible_devices([], "GPU")

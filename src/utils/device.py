from typing import List
from utils.logging import console
import tensorflow as tf


def setup_device(gpu_ids: List[int], memory_limit: float = 0.9) -> None:
    """Configure TensorFlow device and memory settings.

    Args:
        gpu_ids: List of GPU IDs to use. Empty list for CPU.
        memory_limit: Fraction of GPU memory to allocate (0 to 1)
    """
    # Disable eager execution (can help with some CUDA issues)
    # tf.compat.v1.disable_eager_execution()

    console.log("🚀 Setting up devices...")

    if not gpu_ids:
        console.log("💻 Using CPU for training")
        tf.config.set_visible_devices([], "GPU")
        return

    try:
        # Get available GPUs
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            console.log("⚠️ No GPU found. Using CPU instead.")
            return

        # Configure memory growth and limits BEFORE setting visible devices
        for i, gpu in enumerate(gpus):
            try:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)

                # Set memory limit if requested
                if memory_limit < 1.0:
                    # Get total GPU memory in MB
                    gpu_memory = tf.config.experimental.get_memory_info('GPU:0')['current'] / (1024 * 1024)
                    memory_in_mb = int(gpu_memory * memory_limit)
                    tf.config.set_logical_device_configuration(
                        gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_in_mb)]
                    )

                console.log(f"🎮 GPU {i}: {gpu.name}")
                console.log(f"   - Memory limit: {memory_limit*100:.0f}%")

            except RuntimeError as e:
                console.log(f"❌ Error configuring GPU {i}: {e}")
                continue

        # Set visible devices AFTER configuring memory
        visible_gpus = [gpus[i] for i in gpu_ids if i < len(gpus)]
        tf.config.set_visible_devices(visible_gpus, "GPU")
        console.log(f"Found {len(gpus)} GPUs, using {len(visible_gpus)}")

    except RuntimeError as e:
        console.log(f"❌ Error setting up GPU: {e}")
        console.log("⚠️ Falling back to CPU")
        tf.config.set_visible_devices([], "GPU")

    # Final device check
    logical_devices = tf.config.list_logical_devices()
    for device in logical_devices:
        console.log(f"🔍 Available device: {device.name}")

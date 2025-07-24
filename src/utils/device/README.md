# Device Utilities

This module contains utilities for device configuration and management, specifically optimized for JAX backend.

## Files

### 🖥️ `device.py`
Device setup and configuration for JAX backend.

**Functions:**
- `setup_device()` - Configures JAX devices (CPU/GPU) and memory settings

**Features:**
- Automatic GPU detection and configuration
- Memory limit management for GPUs
- CPU fallback support
- Multi-GPU support with device selection
- JAX-specific optimizations

### 📁 `device_configs/`
Reserved directory for device-specific configuration files (currently empty).

## Usage Examples

```python
from src.utils.device.device import setup_device

# Setup GPU with memory limit
setup_device(
    gpu_ids=[0, 1],      # Use GPUs 0 and 1
    memory_limit=0.9     # Use 90% of GPU memory
)

# Setup CPU only
setup_device(
    gpu_ids=[],          # Empty list forces CPU
    memory_limit=1.0
)
```

## JAX Backend Configuration

The device module is optimized for JAX backend:

### Environment Variables Set:
- `JAX_PLATFORM_NAME` - Forces specific platform (cpu/gpu)
- `XLA_PYTHON_CLIENT_MEM_FRACTION` - Controls GPU memory allocation
- `CUDA_VISIBLE_DEVICES` - Specifies which GPUs to use

### Device Information Logged:
- JAX backend type (CPU/GPU/TPU)
- Number of available devices
- Memory configuration
- Device IDs and names

## Configuration Options

From `cfg.device`:
- `gpu_ids` - List of GPU IDs to use (empty for CPU)
- `memory_limit` - Fraction of GPU memory to allocate (0.0-1.0)

## Troubleshooting

**No GPU detected:**
- Falls back to CPU automatically
- Check CUDA installation and drivers
- Verify JAX GPU version is installed

**Out of memory errors:**
- Reduce `memory_limit` parameter
- Use fewer GPUs
- Reduce batch size

**Multi-GPU issues:**
- Ensure all specified GPU IDs exist
- Check CUDA_VISIBLE_DEVICES environment variable
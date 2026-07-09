# Hydra Image Processor (Python)

`hydra-image-processor` provides GPU-accelerated (CUDA) image-processing operations for
1–5 dimensional data (x, y, z, channels, time), efficiently processing data larger than GPU
memory by optimally chunking it.

## Installation

Distributed through [conda-forge](https://conda-forge.org/). Requires an NVIDIA
CUDA-capable GPU with an up-to-date driver.

```bash
conda install -c conda-forge hydra-image-processor
# or: mamba install -c conda-forge hydra-image-processor
# or: pixi add hydra-image-processor
```

> **pip / uv are not supported.** This is a CUDA-only compiled extension with no PyPI
> wheels — install it with conda, mamba, or pixi.

The CUDA runtime is statically linked, so importing the package needs no separate CUDA
toolkit install; an NVIDIA driver is required to run GPU operations.

## Usage

```python
import hydra_image_processor as HIP
import numpy as np

# Check available CUDA devices
print(f"Found {HIP.device_count()} GPU(s)")

# Apply a Gaussian smoothing
image = np.random.rand(100, 100, 50).astype(np.float32)
smoothed = HIP.gaussian(image, sigmas=[2.0, 2.0, 1.0])

# Median filter with a custom kernel
kernel = HIP.make_ball_mask(radius=3)
filtered = HIP.median_filter(image, kernel)
```

### Package layout

- `hydra_image_processor` — the public, Pythonic API (snake_case, e.g. `gaussian`,
  `device_count`). This is the recommended entry point.
- `hydra_image_processor.cuda` — thin wrappers over the compiled `Hydra` CUDA extension.
- `hydra_image_processor.utils` — helpers such as mask creation.

### Backwards compatibility

Earlier releases shipped a top-level `HIP` (and `Hydra`) module exposing the raw extension
API (e.g. `HIP.Gaussian`). Those imports still work via compatibility shims:

```python
import HIP        # legacy: resolves to the compiled extension (raw API)
import Hydra      # legacy: same compiled extension
```

New code should prefer `import hydra_image_processor as HIP`.

## Building from source

Builds use [scikit-build-core](https://scikit-build-core.readthedocs.io/); a single
`pip install` configures CMake, compiles the CUDA extension, and installs the package.
From a conda environment with `cmake`, `ninja`, `scikit-build-core`, and `numpy`, plus a
CUDA toolkit and C++ compiler available:

```bash
# from the repository root (where pyproject.toml lives)
pip install .
```

To build for a specific GPU architecture instead of the default `all-major`:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=native" pip install .
```

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the full developer and release workflow.

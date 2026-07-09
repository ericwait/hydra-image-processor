# Hydra Image Processor (Hydra)

Check out the website at [https://www.hydraimageprocessor.com](https://www.hydraimageprocessor.com)

Hydra Image Processor is a hardware accelerated signal processing library written with [CUDA](https://developer.nvidia.com/cuda-zone).
Hydra aims to create a signal processing library that can be incorporated into many software tools.
This library is licensed under BSD 3-Clause to encourage use in open-source and commercial software.
My only plea is that if you find bugs or make changes that you contribute them back to this repository.
Happy processing and enjoy!

## Installation

Hydra is distributed through [conda-forge](https://conda-forge.org/) and requires an
NVIDIA CUDA-capable GPU with an up-to-date driver. Install it with conda, mamba, or pixi:

```bash
# conda / mamba
conda install -c conda-forge hydra-image-processor
mamba install -c conda-forge hydra-image-processor

# pixi (per-project)
pixi add hydra-image-processor
```

The CUDA runtime is statically linked, so no separate CUDA toolkit is required to *import*
the package — only an NVIDIA GPU driver to *run* GPU operations.

> **pip / uv are not supported.** Hydra is a CUDA-only compiled extension with no PyPI
> wheels. Use a conda-based installer (conda, mamba, or pixi).

### Usage

```python
import hydra_image_processor as HIP
import numpy as np

image = np.random.rand(100, 100, 50).astype(np.float32)
smoothed = HIP.gaussian(image, sigmas=[2.0, 2.0, 1.0])
print("GPUs:", HIP.device_count())
```

Code written against older releases that used `import HIP` (or `import Hydra`) keeps working
via compatibility shims, though `import hydra_image_processor as HIP` is preferred.

### Building from source

Source builds use [scikit-build-core](https://scikit-build-core.readthedocs.io/): a single
`pip install` drives CMake, compiles the CUDA extension, and installs the package. From a
conda environment that has the build tools (`cmake`, `ninja`, `scikit-build-core`, `numpy`)
plus a CUDA toolkit and C++ compiler:

```bash
pip install .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full developer and release workflow.

## Quick Start Guide

[https://www.hydraimageprocessor.com/quick-start](https://www.hydraimageprocessor.com/quick-start)

## Feedback

If you would like to provide feedback about this tutorial or Hydra in general, please use the forum [here](https://www.hydraimageprocessor.com/forum).

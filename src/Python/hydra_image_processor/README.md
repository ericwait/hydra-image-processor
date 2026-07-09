# Hydra Image Processor - Python Package

A Pythonic wrapper for the Hydra Image Processor C++/CUDA library, providing GPU-accelerated image processing operations with automatic CPU fallback.

## Installation

The package can be installed in development mode:

```bash
cd src/Python
pip install -e .
```

## Quick Start

```python
import hydra_image_processor as HIP
import numpy as np

# Check available CUDA devices
num_devices = HIP.device_count()
print(f"Found {num_devices} CUDA device(s)")

# Create a test image
image = np.random.rand(100, 100, 50).astype(np.float32)

# Apply Gaussian smoothing
smoothed = HIP.gaussian(image, sigmas=[2.0, 2.0, 1.0])

# Apply median filter with a ball-shaped kernel
kernel = HIP.make_ball_mask(radius=3)
filtered = HIP.median_filter(image, kernel)

# Get image statistics
min_val, max_val = HIP.get_min_max(image)
total = HIP.sum_array(image)
```

## Package Structure

The package follows a three-layer architecture:

```
hydra_image_processor/
├── __init__.py              # Public API
├── core.py                  # GPU-first fallback logic
├── cuda/                    # GPU/CUDA implementations
│   ├── __init__.py
│   └── core.py             # Wrappers around Hydra.pyd
├── local/                   # CPU fallback implementations (placeholders)
│   ├── __init__.py
│   └── core.py
└── utils/                   # Utility functions
    ├── __init__.py
    └── masks.py            # Mask creation utilities
```

## Features

### Device Management
- `device_count()` - Get number of CUDA devices
- `device_stats()` - Get memory statistics
- `check_config()` - Get library configuration
- `info()` - List all available commands
- `help(command)` - Get help for specific command

### Neighborhood Filters
- `mean_filter()` - Mean/average filtering
- `max_filter()` - Maximum filter (morphological dilation)
- `min_filter()` - Minimum filter (morphological erosion)
- `median_filter()` - Median filtering for noise reduction
- `std_filter()` - Local standard deviation
- `var_filter()` - Local variance

### Gaussian-Based Filters
- `gaussian()` - Gaussian smoothing
- `LoG()` - Laplacian of Gaussian (edge/blob detection)
- `high_pass_filter()` - High-pass filtering

### Morphological Operations
- `closure()` - Morphological closing (dilation → erosion)
- `opener()` - Morphological opening (erosion → dilation)

### Advanced Filters
- `entropy_filter()` - Local entropy for texture analysis
- `wiener_filter()` - Wiener denoising
- `nlmeans()` - Non-Local Means denoising

### Utility Operations
- `multiply_sum()` - Convolution with custom kernel
- `element_wise_difference()` - Element-wise subtraction
- `make_ball_mask()` - Create spherical structuring element
- `make_ellipsoid_mask()` - Create ellipsoidal structuring element

### Reduction Operations
- `sum_array()` - Sum of all array elements
- `get_min_max()` - Get minimum and maximum values

## API Conventions

### Array Dimensions
Unlike MATLAB's `(X, Y, Z, Channel, Time)` convention, this package follows standard Python/NumPy conventions. The specific dimension ordering depends on your use case, but functions accept 1-5D arrays.

### Naming Conventions
- Functions use `snake_case` (Pythonic style)
- Acronyms preserve their case (e.g., `LoG` not `log`)
- The conventional import alias is `HIP`: `import hydra_image_processor as HIP`

### Device Selection
All processing functions accept an optional `device` parameter:
- `device=None` (default): Automatically selects best device(s)
- `device=-1`: Explicitly use all available GPUs
- `device=0, 1, 2, ...`: Use specific GPU device

```python
# Let the library choose
result = HIP.gaussian(image, sigmas=[1, 1, 1])

# Use all GPUs explicitly
result = HIP.gaussian(image, sigmas=[1, 1, 1], device=-1)

# Use specific GPU
result = HIP.gaussian(image, sigmas=[1, 1, 1], device=0)
```

### GPU-First Fallback
The package automatically attempts GPU acceleration first, falling back to CPU implementations if GPU is unavailable. Currently, CPU implementations are placeholders and will raise `NotImplementedError`.

## Documentation Style

The package uses NumPy-style docstrings, which are compatible with Sphinx and provide clear, structured documentation. Access documentation via:

```python
# Python's built-in help
help(HIP.gaussian)

# Hydra's help system (from C++ layer)
HIP.help('Gaussian')

# IPython/Jupyter
HIP.gaussian?
```

## Examples

### Example 1: Basic Filtering

```python
import hydra_image_processor as HIP
import numpy as np

# Create test image
image = np.random.rand(128, 128, 64).astype(np.float32)

# Apply Gaussian blur
blurred = HIP.gaussian(image, sigmas=[3.0, 3.0, 1.5])

# Apply median filter for noise reduction
kernel = HIP.make_ball_mask(radius=2)
denoised = HIP.median_filter(image, kernel)
```

### Example 2: Morphological Operations

```python
import hydra_image_processor as HIP
import numpy as np

# Binary image
binary = (np.random.rand(100, 100, 50) > 0.5).astype(np.float32)

# Create structuring element
se = HIP.make_ball_mask(radius=3)

# Morphological closing (fill small holes)
closed = HIP.closure(binary, se)

# Morphological opening (remove small objects)
opened = HIP.opener(binary, se)
```

### Example 3: Edge Detection

```python
import hydra_image_processor as HIP
import numpy as np

# Load or create image
image = np.random.rand(256, 256).astype(np.float32)

# Detect edges using Laplacian of Gaussian
edges = HIP.LoG(image, sigmas=[2.0, 2.0, 0.0])

# Or use high-pass filtering
high_freq = HIP.high_pass_filter(image, sigmas=[3.0, 3.0, 0.0])
```

## Requirements

- Python >= 3.9
- NumPy
- Hydra.pyd (compiled C++/CUDA extension)
- CUDA-capable GPU (for GPU acceleration)
- OpenMP DLL (libomp140.x86_64.dll on Windows)

## Contributing

CPU fallback implementations are currently placeholders. Contributions of NumPy/SciPy-based CPU implementations are welcome! The MATLAB package in `src/MATLAB/+HIP/+Local` provides reference implementations that can be translated to Python.

## License

BSD-3-Clause (same as the main Hydra Image Processor project)

## Authors

- Python Package: Eric Wait
- Original C++/CUDA Library: Eric Wait

## See Also

- Main Project: [Hydra Image Processor](../../)
- MATLAB Package: [src/MATLAB/+HIP](../../MATLAB/+HIP)
- C++ Source: [src/c](../../c)

"""
Hydra Image Processor - High-Performance GPU-Accelerated Image Processing

A Python package providing GPU-accelerated image processing operations with
automatic CPU fallback. Built on top of the Hydra C++/CUDA library.

Basic Usage
-----------
Import the package with the conventional alias::

    import hydra_image_processor as HIP

Check available CUDA devices::

    num_devices = HIP.device_count()
    print(f"Found {num_devices} GPU(s)")

Apply filters to images::

    import numpy as np

    # Create test image
    image = np.random.rand(100, 100, 50).astype(np.float32)

    # Apply Gaussian smoothing
    smoothed = HIP.gaussian(image, sigmas=[2.0, 2.0, 1.0])

    # Apply median filter with custom kernel
    kernel = HIP.make_ball_mask(radius=3)
    filtered = HIP.median_filter(image, kernel)

GPU Management
--------------
By default, functions automatically select the best available GPU or split
work across multiple GPUs. You can explicitly control device selection::

    # Use specific GPU
    result = HIP.gaussian(image, sigmas=[1, 1, 1], device=0)

    # Explicitly use all GPUs
    result = HIP.gaussian(image, sigmas=[1, 1, 1], device=-1)

Package Organization
--------------------
- `hydra_image_processor.cuda`: Direct GPU/CUDA implementations
- `hydra_image_processor.local`: CPU fallback implementations (stubs)
- `hydra_image_processor.utils`: Utility functions (mask creation, etc.)

Available Functions
-------------------
**Device Management**
    - device_count, device_stats, check_config, info, help

**Neighborhood Filters**
    - mean_filter, max_filter, min_filter, median_filter, std_filter, var_filter

**Gaussian-Based Filters**
    - gaussian, LoG, high_pass_filter

**Morphological Operations**
    - closure, opener

**Advanced Filters**
    - entropy_filter, wiener_filter, nlmeans

**Utility Operations**
    - multiply_sum, element_wise_difference

**Reduction Operations**
    - sum_array, get_min_max

**Utilities**
    - make_ball_mask, make_ellipsoid_mask

**Test/Debug**
    - identity_filter
"""

__version__ = "0.1.0"
__author__ = "Eric Wait"
__email__ = "info@ericwait.com"

# Import public API
from .core import (
    # Device management
    device_count,
    device_stats,
    check_config,
    info,
    help,

    # Neighborhood filters
    mean_filter,
    max_filter,
    min_filter,
    median_filter,
    std_filter,
    var_filter,

    # Gaussian-based filters
    gaussian,
    LoG,
    high_pass_filter,

    # Morphological operations
    closure,
    opener,

    # Advanced filters
    entropy_filter,
    wiener_filter,
    nlmeans,

    # Utility operations
    multiply_sum,
    element_wise_difference,

    # Reduction operations
    sum_array,
    get_min_max,

    # Identity/test
    identity_filter,
)

# Import utilities
from .utils import (
    make_ball_mask,
    make_ellipsoid_mask,
)

# Define public API
__all__ = [
    # Package metadata
    '__version__',
    '__author__',
    '__email__',

    # Device management
    'device_count',
    'device_stats',
    'check_config',
    'info',
    'help',

    # Neighborhood filters
    'mean_filter',
    'max_filter',
    'min_filter',
    'median_filter',
    'std_filter',
    'var_filter',

    # Gaussian-based filters
    'gaussian',
    'LoG',
    'high_pass_filter',

    # Morphological operations
    'closure',
    'opener',

    # Advanced filters
    'entropy_filter',
    'wiener_filter',
    'nlmeans',

    # Utility operations
    'multiply_sum',
    'element_wise_difference',

    # Reduction operations
    'sum_array',
    'get_min_max',

    # Utilities
    'make_ball_mask',
    'make_ellipsoid_mask',

    # Identity/test
    'identity_filter',
]

"""
Local/CPU implementation layer for Hydra Image Processor.

This module provides CPU-based fallback implementations for image processing
operations. Currently contains placeholder stubs that will be implemented
with NumPy/SciPy in the future.
"""

from .core import (
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
    sum as sum_array,
    get_min_max,

    # Identity/test
    identity_filter,
)

__all__ = [
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

    # Identity/test
    'identity_filter',
]

"""
CUDA/GPU implementation layer for Hydra Image Processor.

This module provides thin wrappers around the compiled Hydra C++ extension,
enabling GPU-accelerated image processing operations.
"""

from .core import (
    # Device management
    device_count,
    device_stats,
    check_config,
    info,
    help_func,

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
    # Device management
    'device_count',
    'device_stats',
    'check_config',
    'info',
    'help_func',

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

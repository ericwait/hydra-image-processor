"""
Core public API for Hydra Image Processor.

This module provides the main user-facing API with automatic fallback from
GPU (CUDA) to CPU implementations. Functions attempt to use GPU acceleration
first, falling back to CPU implementations if GPU is unavailable.
"""

import warnings
import numpy as np
from typing import Optional, Union, List, Tuple
from functools import wraps

from . import cuda
from . import local


def _gpu_with_fallback(cuda_func, local_func):
    """
    Decorator factory that creates a function with GPU-first fallback logic.

    Attempts to call the CUDA implementation first. If it fails (due to no GPU,
    driver issues, etc.), warns the user and falls back to the CPU implementation.

    Parameters
    ----------
    cuda_func : callable
        GPU implementation function.
    local_func : callable
        CPU fallback implementation function.

    Returns
    -------
    callable
        Wrapped function with fallback logic.
    """
    @wraps(cuda_func)
    def wrapper(*args, **kwargs):
        try:
            return cuda_func(*args, **kwargs)
        except Exception as e:
            # Check if it's the "not yet implemented" error from CPU version
            # to avoid double warnings
            warnings.warn(
                f"GPU implementation failed: {str(e)}. "
                f"Falling back to CPU implementation.",
                RuntimeWarning,
                stacklevel=2
            )
            return local_func(*args, **kwargs)

    return wrapper


# ==============================================================================
# Device Management Functions
# ==============================================================================

def device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns
    -------
    int
        Number of CUDA-capable devices available on the system.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.

    Examples
    --------
    >>> import hydra_image_processor as HIP
    >>> num_devices = HIP.device_count()
    >>> print(f"Found {num_devices} CUDA device(s)")
    """
    return cuda.device_count()


def device_stats() -> List[dict]:
    """
    Get memory statistics for all CUDA devices.

    Returns
    -------
    List[dict]
        List of dictionaries containing memory statistics for each device.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.

    Examples
    --------
    >>> import hydra_image_processor as HIP
    >>> stats = HIP.device_stats()
    >>> for i, stat in enumerate(stats):
    ...     print(f"Device {i}: {stat}")
    """
    return cuda.device_stats()


def check_config() -> dict:
    """
    Get Hydra library configuration information.

    Returns
    -------
    dict
        Configuration dictionary containing library build information,
        CUDA capabilities, and other system details.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.

    Examples
    --------
    >>> import hydra_image_processor as HIP
    >>> config = HIP.check_config()
    >>> print("Library configuration:", config)
    """
    return cuda.check_config()


def info() -> List[dict]:
    """
    Get information about all available Hydra commands.

    Returns
    -------
    List[dict]
        List of dictionaries, each containing:
        - 'command': Command name
        - 'inArgs': Comma-separated input arguments
        - 'outArgs': Comma-separated output arguments
        - 'help': Detailed help text for the command

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.

    Examples
    --------
    >>> import hydra_image_processor as HIP
    >>> commands = HIP.info()
    >>> print(f"Available commands: {len(commands)}")
    >>> for cmd in commands[:3]:
    ...     print(f"- {cmd['command']}: {cmd['inArgs']} -> {cmd['outArgs']}")
    """
    return cuda.info()


def help(command: Optional[str] = None) -> str:
    """
    Get help text for a specific Hydra command.

    Parameters
    ----------
    command : str, optional
        Name of the command to get help for. If None, returns general help.

    Returns
    -------
    str
        Help text for the specified command.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.

    Examples
    --------
    >>> import hydra_image_processor as HIP
    >>> print(HIP.help('Gaussian'))
    """
    return cuda.help_func(command)


# ==============================================================================
# Neighborhood Filter Functions
# ==============================================================================

mean_filter = _gpu_with_fallback(cuda.mean_filter, local.mean_filter)
mean_filter.__doc__ = cuda.mean_filter.__doc__

max_filter = _gpu_with_fallback(cuda.max_filter, local.max_filter)
max_filter.__doc__ = cuda.max_filter.__doc__

min_filter = _gpu_with_fallback(cuda.min_filter, local.min_filter)
min_filter.__doc__ = cuda.min_filter.__doc__

median_filter = _gpu_with_fallback(cuda.median_filter, local.median_filter)
median_filter.__doc__ = cuda.median_filter.__doc__

std_filter = _gpu_with_fallback(cuda.std_filter, local.std_filter)
std_filter.__doc__ = cuda.std_filter.__doc__

var_filter = _gpu_with_fallback(cuda.var_filter, local.var_filter)
var_filter.__doc__ = cuda.var_filter.__doc__


# ==============================================================================
# Gaussian-Based Filter Functions
# ==============================================================================

gaussian = _gpu_with_fallback(cuda.gaussian, local.gaussian)
gaussian.__doc__ = cuda.gaussian.__doc__

LoG = _gpu_with_fallback(cuda.LoG, local.LoG)
LoG.__doc__ = cuda.LoG.__doc__

high_pass_filter = _gpu_with_fallback(cuda.high_pass_filter, local.high_pass_filter)
high_pass_filter.__doc__ = cuda.high_pass_filter.__doc__


# ==============================================================================
# Morphological Operations
# ==============================================================================

closure = _gpu_with_fallback(cuda.closure, local.closure)
closure.__doc__ = cuda.closure.__doc__

opener = _gpu_with_fallback(cuda.opener, local.opener)
opener.__doc__ = cuda.opener.__doc__


# ==============================================================================
# Advanced Filter Functions
# ==============================================================================

entropy_filter = _gpu_with_fallback(cuda.entropy_filter, local.entropy_filter)
entropy_filter.__doc__ = cuda.entropy_filter.__doc__

wiener_filter = _gpu_with_fallback(cuda.wiener_filter, local.wiener_filter)
wiener_filter.__doc__ = cuda.wiener_filter.__doc__

nlmeans = _gpu_with_fallback(cuda.nlmeans, local.nlmeans)
nlmeans.__doc__ = cuda.nlmeans.__doc__


# ==============================================================================
# Utility Operations
# ==============================================================================

multiply_sum = _gpu_with_fallback(cuda.multiply_sum, local.multiply_sum)
multiply_sum.__doc__ = cuda.multiply_sum.__doc__

element_wise_difference = _gpu_with_fallback(
    cuda.element_wise_difference,
    local.element_wise_difference
)
element_wise_difference.__doc__ = cuda.element_wise_difference.__doc__


# ==============================================================================
# Reduction Operations
# ==============================================================================

sum_array = _gpu_with_fallback(cuda.sum_array, local.sum_array)
sum_array.__doc__ = cuda.sum_array.__doc__

get_min_max = _gpu_with_fallback(cuda.get_min_max, local.get_min_max)
get_min_max.__doc__ = cuda.get_min_max.__doc__


# ==============================================================================
# Identity/Test Functions
# ==============================================================================

identity_filter = _gpu_with_fallback(cuda.identity_filter, local.identity_filter)
identity_filter.__doc__ = cuda.identity_filter.__doc__

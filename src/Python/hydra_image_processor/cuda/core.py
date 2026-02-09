"""
Core CUDA wrapper functions for Hydra Image Processor.

This module provides thin wrappers around the Hydra.pyd C++ extension module,
converting between Pythonic naming conventions and the underlying C++ API.
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Any
import warnings

# Try to import the compiled Hydra module from parent package
try:
    from .. import Hydra as _Hydra
    _HYDRA_AVAILABLE = True
except ImportError as e:
    _HYDRA_AVAILABLE = False
    _HYDRA_IMPORT_ERROR = str(e)


def _ensure_hydra_available():
    """Raise an error if Hydra module is not available."""
    if not _HYDRA_AVAILABLE:
        raise ImportError(
            f"Hydra C++ extension module is not available: {_HYDRA_IMPORT_ERROR}. "
            "Make sure Hydra.pyd is in the Python path along with required DLLs."
        )


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
    """
    _ensure_hydra_available()
    result = _Hydra.DeviceCount()
    # DeviceCount returns (count, stats), extract just the count
    if isinstance(result, tuple):
        return result[0]
    return result


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
    """
    _ensure_hydra_available()
    result = _Hydra.DeviceCount()
    # DeviceCount returns (count, stats), extract the stats
    if isinstance(result, tuple):
        return result[1]
    # Fallback to DeviceStats if it exists separately
    return _Hydra.DeviceStats()


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
    """
    _ensure_hydra_available()
    return _Hydra.CheckConfig()


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
    """
    _ensure_hydra_available()
    return _Hydra.Info()


def help_func(command: Optional[str] = None) -> str:
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
    """
    _ensure_hydra_available()
    if command is None:
        return _Hydra.Help()
    return _Hydra.Help(command)


# ==============================================================================
# Neighborhood Filter Functions
# ==============================================================================

def mean_filter(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    num_iterations: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply mean (average) filter to image using the specified kernel.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions). First 3 dimensions are spatial,
        4th is channel, 5th is time/frame.
    kernel : array-like
        Kernel array (1-3 dimensions) defining the neighborhood. Non-zero
        elements indicate positions to include in the mean calculation.
    num_iterations : int, default=1
        Number of times to apply the filter.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Filtered image with same shape and dtype as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.MeanFilter(image, kernel, num_iterations, device)


def max_filter(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    num_iterations: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply maximum filter to image (morphological dilation).

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    kernel : array-like
        Structuring element for dilation. Non-zero elements define the
        neighborhood shape.
    num_iterations : int, default=1
        Number of times to apply the filter.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Filtered image with same shape and dtype as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.MaxFilter(image, kernel, num_iterations, device)


def min_filter(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    num_iterations: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply minimum filter to image (morphological erosion).

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    kernel : array-like
        Structuring element for erosion. Non-zero elements define the
        neighborhood shape.
    num_iterations : int, default=1
        Number of times to apply the filter.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Filtered image with same shape and dtype as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.MinFilter(image, kernel, num_iterations, device)


def median_filter(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    num_iterations: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply median filter to image for noise reduction.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    kernel : array-like
        Kernel defining the neighborhood shape. Non-zero elements indicate
        positions to include in median calculation.
    num_iterations : int, default=1
        Number of times to apply the filter.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Filtered image with same shape and dtype as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.MedianFilter(image, kernel, num_iterations, device)


def std_filter(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    num_iterations: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Calculate standard deviation within local neighborhoods.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    kernel : array-like
        Kernel defining the neighborhood shape.
    num_iterations : int, default=1
        Number of times to apply the filter.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Image containing local standard deviations, same shape as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.StdFilter(image, kernel, num_iterations, device)


def var_filter(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    num_iterations: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Calculate variance within local neighborhoods.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    kernel : array-like
        Kernel defining the neighborhood shape.
    num_iterations : int, default=1
        Number of times to apply the filter.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Image containing local variances, same shape as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.VarFilter(image, kernel, num_iterations, device)


# ==============================================================================
# Gaussian-Based Filter Functions
# ==============================================================================

def gaussian(
    image: np.ndarray,
    sigmas: Union[List[float], Tuple[float, ...], np.ndarray],
    num_iterations: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply Gaussian smoothing filter to image.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    sigmas : array-like of float
        Gaussian sigma values [sigma_x, sigma_y, sigma_z]. Use 0 for no
        smoothing in that dimension.
    num_iterations : int, default=1
        Number of times to apply the filter.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Smoothed image with same shape and dtype as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    sigmas = np.asarray(sigmas, dtype=np.float32)
    if device is None:
        device = -1
    return _Hydra.Gaussian(image, sigmas, num_iterations, device)


def LoG(
    image: np.ndarray,
    sigmas: Union[List[float], Tuple[float, ...], np.ndarray],
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply Laplacian of Gaussian (LoG) filter for edge/blob detection.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    sigmas : array-like of float
        Gaussian sigma values [sigma_x, sigma_y, sigma_z] for the LoG kernel.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        LoG-filtered image with same shape as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    sigmas = np.asarray(sigmas, dtype=np.float32)
    if device is None:
        device = -1
    return _Hydra.LoG(image, sigmas, device)


def high_pass_filter(
    image: np.ndarray,
    sigmas: Union[List[float], Tuple[float, ...], np.ndarray],
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply high-pass filter to enhance high-frequency details.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    sigmas : array-like of float
        Gaussian sigma values for the high-pass filter.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        High-pass filtered image with same shape as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    sigmas = np.asarray(sigmas, dtype=np.float32)
    if device is None:
        device = -1
    return _Hydra.HighPassFilter(image, sigmas, device)


# ==============================================================================
# Morphological Operations
# ==============================================================================

def closure(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    num_iterations: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply morphological closing (dilation followed by erosion).

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    kernel : array-like
        Structuring element. Non-zero elements define the neighborhood shape.
    num_iterations : int, default=1
        Number of times to apply the operation.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Morphologically closed image with same shape and dtype as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.Closure(image, kernel, num_iterations, device)


def opener(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    num_iterations: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply morphological opening (erosion followed by dilation).

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    kernel : array-like
        Structuring element. Non-zero elements define the neighborhood shape.
    num_iterations : int, default=1
        Number of times to apply the operation.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Morphologically opened image with same shape and dtype as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.Opener(image, kernel, num_iterations, device)


# ==============================================================================
# Advanced Filter Functions
# ==============================================================================

def entropy_filter(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    device: Optional[int] = None
) -> np.ndarray:
    """
    Calculate local entropy within neighborhoods for texture analysis.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    kernel : array-like
        Kernel defining the neighborhood shape.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Image containing local entropy values, same shape as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.EntropyFilter(image, kernel, device)


def wiener_filter(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    noise_variance: float,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply Wiener filter for noise reduction.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    kernel : array-like
        Kernel defining the neighborhood shape.
    noise_variance : float
        Estimated variance of the noise in the image.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Wiener-filtered image with same shape as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.WienerFilter(image, kernel, noise_variance, device)


def nlmeans(
    image: np.ndarray,
    h: float,
    search_window_radius: int,
    neighborhood_radius: int,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply Non-Local Means denoising filter.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    h : float
        Filtering parameter controlling decay. Higher values remove more noise
        but may blur details.
    search_window_radius : int
        Radius of the search window for finding similar patches.
    neighborhood_radius : int
        Radius of the neighborhood/patch for comparison.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Denoised image with same shape as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    if device is None:
        device = -1
    return _Hydra.NLMeans(image, h, search_window_radius, neighborhood_radius, device)


# ==============================================================================
# Utility Operations
# ==============================================================================

def multiply_sum(
    image: np.ndarray,
    kernel: Union[np.ndarray, List[int], Tuple[int, ...]],
    num_iterations: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Apply weighted sum (convolution) using the specified kernel.

    Parameters
    ----------
    image : np.ndarray
        Input image array (1-5 dimensions).
    kernel : array-like
        Convolution kernel with weights for each neighborhood position.
    num_iterations : int, default=1
        Number of times to apply the convolution.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Convolved image with same shape as input.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    kernel = np.asarray(kernel)
    if device is None:
        device = -1
    return _Hydra.MultiplySum(image, kernel, num_iterations, device)


def element_wise_difference(
    image1: np.ndarray,
    image2: np.ndarray,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Compute element-wise difference between two images (image1 - image2).

    Parameters
    ----------
    image1 : np.ndarray
        First input image.
    image2 : np.ndarray
        Second input image (subtracted from first).
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Difference image with same shape as inputs.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    if device is None:
        device = -1
    return _Hydra.ElementWiseDifference(image1, image2, device)


# ==============================================================================
# Reduction Operations
# ==============================================================================

def sum(
    image: np.ndarray,
    device: Optional[int] = None
) -> float:
    """
    Compute the sum of all elements in the array.

    Parameters
    ----------
    image : np.ndarray
        Input image array.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    float
        Sum of all array elements.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    if device is None:
        device = -1
    return _Hydra.Sum(image, device)


def get_min_max(
    image: np.ndarray,
    device: Optional[int] = None
) -> Tuple[float, float]:
    """
    Get minimum and maximum values in the array.

    Parameters
    ----------
    image : np.ndarray
        Input image array.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    Tuple[float, float]
        Tuple containing (min_value, max_value).

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    if device is None:
        device = -1
    return _Hydra.GetMinMax(image, device)


# ==============================================================================
# Identity/Test Functions
# ==============================================================================

def identity_filter(
    image: np.ndarray,
    device: Optional[int] = None
) -> np.ndarray:
    """
    Identity operation (returns copy of input). Useful for testing.

    Parameters
    ----------
    image : np.ndarray
        Input image array.
    device : int, optional
        CUDA device ID to use. If None (default), automatically selects device
        or splits across multiple devices. Use -1 to explicitly use all GPUs.

    Returns
    -------
    np.ndarray
        Copy of input image.

    Raises
    ------
    ImportError
        If the Hydra C++ extension is not available.
    """
    _ensure_hydra_available()
    if device is None:
        device = -1
    return _Hydra.IdentityFilter(image, device)

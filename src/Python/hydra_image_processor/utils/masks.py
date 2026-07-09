"""
Mask and structuring element creation utilities.

This module provides functions to create common structuring elements and masks
for use with morphological operations and neighborhood filters.
"""

import numpy as np
from typing import Union, Tuple, List


def make_ball_mask(radius: int) -> np.ndarray:
    """
    Create a 3D spherical (ball) structuring element.

    Creates a boolean array where True values form a sphere of the specified
    radius. This is useful for morphological operations and neighborhood filters
    with spherical neighborhoods.

    Parameters
    ----------
    radius : int
        Radius of the ball in pixels. Must be positive.

    Returns
    -------
    np.ndarray
        3D boolean array of shape (2*radius+1, 2*radius+1, 2*radius+1) where
        True values indicate positions inside the sphere.

    Examples
    --------
    >>> import hydra_image_processor as HIP
    >>> mask = HIP.make_ball_mask(3)
    >>> mask.shape
    (7, 7, 7)
    >>> mask.sum()  # Number of voxels in the ball
    123

    Notes
    -----
    The ball is centered in the array, with radius measured from the center
    voxel. The function uses Euclidean distance to determine inclusion.
    """
    if radius <= 0:
        raise ValueError("Radius must be positive")

    size = 2 * radius + 1
    shape_element = np.zeros((size, size, size), dtype=bool)

    # Create coordinate grids
    x, y, z = np.mgrid[0:size, 0:size, 0:size]

    # Calculate distance from center
    center = radius
    distances_squared = (
        (x - center) ** 2 +
        (y - center) ** 2 +
        (z - center) ** 2
    )

    # Set True for positions inside the ball
    shape_element[distances_squared <= radius ** 2] = True

    return shape_element


def make_ellipsoid_mask(
    axes_radius: Union[Tuple[float, float, float], List[float], np.ndarray]
) -> np.ndarray:
    """
    Create a 3D ellipsoidal structuring element.

    Creates a boolean array where True values form an ellipsoid with the
    specified radii along each axis. This is useful for anisotropic
    morphological operations and neighborhood filters.

    Parameters
    ----------
    axes_radius : array-like of 3 floats
        Radii along each axis [radius_x, radius_y, radius_z]. Values must be
        non-negative. Use 0 for a flat disk in that dimension.

    Returns
    -------
    np.ndarray
        3D boolean array where True values indicate positions inside the
        ellipsoid. Array size is determined by the radii to fully contain
        the ellipsoid.

    Raises
    ------
    ValueError
        If axes_radius does not contain exactly 3 values or contains
        negative values.

    Examples
    --------
    >>> import hydra_image_processor as HIP
    >>> # Create an ellipsoid with different radii
    >>> mask = HIP.make_ellipsoid_mask([5, 3, 2])
    >>> mask.shape
    (11, 7, 5)

    >>> # Create a disk (flat in Z)
    >>> disk = HIP.make_ellipsoid_mask([5, 5, 0])
    >>> disk.shape[2]
    1

    Notes
    -----
    The ellipsoid is centered in the array. The inclusion criterion uses
    the standard ellipsoid equation: (x/rx)² + (y/ry)² + (z/rz)² ≤ 1
    """
    axes_radius = np.asarray(axes_radius, dtype=float)

    if axes_radius.size != 3:
        raise ValueError(
            f"axes_radius must contain exactly 3 values (got {axes_radius.size}). "
            "Specify radii for X, Y, and Z axes."
        )

    if np.any(axes_radius < 0):
        raise ValueError("All radius values must be non-negative")

    # Calculate volume size (ensure at least size 1 in each dimension)
    vol_size = np.maximum(np.ceil(axes_radius * 2) + 1, 1).astype(int)

    shape_element = np.zeros(vol_size, dtype=bool)

    # Create coordinate grids
    x, y, z = np.mgrid[0:vol_size[0], 0:vol_size[1], 0:vol_size[2]]

    # Shift origin to the center voxel, matching the C++ createEllipsoidKernel
    # (for odd sizes the center lands on an integer index)
    x = x - (vol_size[0] - 1) / 2
    y = y - (vol_size[1] - 1) / 2
    z = z - (vol_size[2] - 1) / 2

    # Avoid division by zero for zero radii
    axes_radius = np.maximum(axes_radius, 1e-10)

    # Calculate normalized squared distances
    normalized_distances = (
        (x ** 2 / axes_radius[0] ** 2) +
        (y ** 2 / axes_radius[1] ** 2) +
        (z ** 2 / axes_radius[2] ** 2)
    )

    # Set True for positions inside the ellipsoid
    shape_element[normalized_distances <= 1] = True

    return shape_element

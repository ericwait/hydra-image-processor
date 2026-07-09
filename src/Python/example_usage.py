#!/usr/bin/env python3
"""
Example usage of the hydra_image_processor package.

This script demonstrates various image processing operations available
in the Hydra Image Processor Python package.
"""

import sys
import numpy as np

# Add package to path for development
sys.path.insert(0, '.')

import hydra_image_processor as HIP


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def example_device_info():
    """Display information about available CUDA devices."""
    print_section("Device Information")

    count = HIP.device_count()
    print(f"Number of CUDA devices: {count}")

    if count > 0:
        stats = HIP.device_stats()
        for i, stat in enumerate(stats):
            total_gb = stat['total'] / (1024**3)
            avail_gb = stat['available'] / (1024**3)
            print(f"\nDevice {i}:")
            print(f"  Total memory: {total_gb:.2f} GB")
            print(f"  Available memory: {avail_gb:.2f} GB")


def example_basic_filtering():
    """Demonstrate basic filtering operations."""
    print_section("Basic Filtering Operations")

    # Create a test image
    print("\nCreating test image (100x100x50)...")
    image = np.random.rand(100, 100, 50).astype(np.float32)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    # Gaussian smoothing
    print("\n1. Applying Gaussian smoothing (sigma=2.0)...")
    smoothed = HIP.gaussian(image, sigmas=[2.0, 2.0, 1.0])
    print(f"   Output shape: {smoothed.shape}")

    # Mean filter
    print("\n2. Applying mean filter (3x3x3 kernel)...")
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    mean_result = HIP.mean_filter(image, kernel)
    print(f"   Output shape: {mean_result.shape}")

    # Median filter
    print("\n3. Applying median filter with ball mask (radius=2)...")
    ball = HIP.make_ball_mask(radius=2)
    print(f"   Ball mask shape: {ball.shape}, active voxels: {ball.sum()}")
    median_result = HIP.median_filter(image, ball)
    print(f"   Output shape: {median_result.shape}")


def example_morphology():
    """Demonstrate morphological operations."""
    print_section("Morphological Operations")

    # Create binary image
    print("\nCreating binary test image (80x80x40)...")
    binary = (np.random.rand(80, 80, 40) > 0.6).astype(np.float32)
    print(f"Image shape: {binary.shape}")
    print(f"Foreground voxels: {binary.sum():.0f} ({100*binary.mean():.1f}%)")

    # Create structuring element
    print("\nCreating ellipsoid structuring element...")
    se = HIP.make_ellipsoid_mask([3, 3, 2])
    print(f"SE shape: {se.shape}, active elements: {se.sum()}")

    # Morphological closing
    print("\nApplying morphological closing...")
    closed = HIP.closure(binary, se)
    print(f"Result shape: {closed.shape}")
    print(f"Foreground after closing: {closed.sum():.0f} ({100*closed.mean():.1f}%)")

    # Morphological opening
    print("\nApplying morphological opening...")
    opened = HIP.opener(binary, se)
    print(f"Result shape: {opened.shape}")
    print(f"Foreground after opening: {opened.sum():.0f} ({100*opened.mean():.1f}%)")


def example_edge_detection():
    """Demonstrate edge detection operations."""
    print_section("Edge Detection")

    # Create test image with some structure
    print("\nCreating structured test image (128x128)...")
    x = np.linspace(-5, 5, 128)
    y = np.linspace(-5, 5, 128)
    X, Y = np.meshgrid(x, y)
    image = (np.sin(X) * np.cos(Y)).astype(np.float32)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    # Laplacian of Gaussian
    print("\nApplying Laplacian of Gaussian (sigma=2.0)...")
    edges = HIP.LoG(image, sigmas=[2.0, 2.0, 0.0])
    print(f"Edge map shape: {edges.shape}")

    # High-pass filter
    print("\nApplying high-pass filter (sigma=3.0)...")
    high_freq = HIP.high_pass_filter(image, sigmas=[3.0, 3.0, 0.0])
    print(f"High-freq shape: {high_freq.shape}")


def example_statistics():
    """Demonstrate statistical operations."""
    print_section("Image Statistics")

    # Create test image
    print("\nCreating test image...")
    image = np.random.rand(50, 50, 25).astype(np.float32) * 100
    print(f"Image shape: {image.shape}")

    # Get min/max
    min_val, max_val = HIP.get_min_max(image)
    print(f"\nMin value: {min_val:.4f}")
    print(f"Max value: {max_val:.4f}")

    # Get sum
    total = HIP.sum_array(image)
    mean_val = total / image.size
    print(f"\nSum: {total:.2f}")
    print(f"Mean (calculated): {mean_val:.4f}")

    # Standard deviation filter
    print("\nApplying std filter (5x5x5 kernel)...")
    kernel = np.ones((5, 5, 5), dtype=np.uint8)
    std_result = HIP.std_filter(image, kernel)
    local_std_mean = HIP.sum_array(std_result) / std_result.size
    print(f"Mean local std: {local_std_mean:.4f}")


def example_mask_creation():
    """Demonstrate mask/structuring element creation."""
    print_section("Mask Creation Utilities")

    # Ball mask
    print("\nCreating ball masks of various radii...")
    for radius in [2, 5, 10]:
        ball = HIP.make_ball_mask(radius)
        expected_size = 2 * radius + 1
        print(f"  Radius {radius}: shape={ball.shape}, "
              f"voxels={ball.sum()}, expected_dim={expected_size}")

    # Ellipsoid masks
    print("\nCreating ellipsoid masks with different aspect ratios...")
    configs = [
        ([5, 5, 5], "Sphere"),
        ([10, 5, 2], "Elongated"),
        ([5, 5, 0], "Disk (flat in Z)"),
    ]

    for axes, description in configs:
        mask = HIP.make_ellipsoid_mask(axes)
        print(f"  {description} {axes}: shape={mask.shape}, voxels={mask.sum()}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print(" " * 15 + "Hydra Image Processor")
    print(" " * 19 + "Example Usage")
    print("=" * 60)

    try:
        example_device_info()
        example_mask_creation()
        example_basic_filtering()
        example_morphology()
        example_edge_detection()
        example_statistics()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

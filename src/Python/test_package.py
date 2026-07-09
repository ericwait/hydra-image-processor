"""
Test script for the hydra_image_processor package.

This script demonstrates basic usage and tests key functionality of the
Pythonic wrapper around Hydra Image Processor.
"""

import sys
import numpy as np

# Add current directory to path for testing
sys.path.insert(0, '.')

import hydra_image_processor as HIP


def test_import():
    """Test that the package imports correctly."""
    print("=" * 60)
    print("Test 1: Package Import")
    print("=" * 60)
    print("[OK] Package imported successfully")
    print(f"  Version: {HIP.__version__}")
    print(f"  Author: {HIP.__author__}")
    print(f"  Available functions: {len(HIP.__all__)}")
    print()
    return True


def test_device_info():
    """Test device information functions."""
    print("=" * 60)
    print("Test 2: Device Information")
    print("=" * 60)
    try:
        count = HIP.device_count()
        print(f"[OK] Found {count} CUDA device(s)")

        if count > 0:
            stats = HIP.device_stats()
            print(f"  Device statistics available for {len(stats)} device(s)")

        config = HIP.check_config()
        print("  Library configuration retrieved")

        commands = HIP.info()
        print(f"  Available commands: {len(commands)}")

        return True
    except Exception as e:
        print(f"[FAILED] Device info test failed: {e}")
        return False
    finally:
        print()


def test_utility_functions():
    """Test utility functions like mask creation."""
    print("=" * 60)
    print("Test 3: Utility Functions")
    print("=" * 60)
    try:
        # Test ball mask creation
        ball = HIP.make_ball_mask(radius=5)
        print(f"[OK] Created ball mask: shape={ball.shape}, dtype={ball.dtype}")
        print(f"  Voxels in ball: {ball.sum()}")

        # Test ellipsoid mask creation
        ellipsoid = HIP.make_ellipsoid_mask([5, 3, 2])
        print(f"[OK] Created ellipsoid mask: shape={ellipsoid.shape}, dtype={ellipsoid.dtype}")
        print(f"  Voxels in ellipsoid: {ellipsoid.sum()}")

        return True
    except Exception as e:
        print(f"[FAILED] Utility functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print()


def test_image_processing():
    """Test basic image processing operations."""
    print("=" * 60)
    print("Test 4: Image Processing Operations")
    print("=" * 60)
    try:
        # Create test image
        image = np.random.rand(50, 50, 20).astype(np.float32)
        print(f"Created test image: shape={image.shape}, dtype={image.dtype}")

        # Test Gaussian filtering
        print("Testing Gaussian filter...")
        smoothed = HIP.gaussian(image, sigmas=[2.0, 2.0, 1.0])
        print(f"[OK] Gaussian filter: input {image.shape} -> output {smoothed.shape}")

        # Test mean filter with small kernel
        print("Testing mean filter...")
        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        mean_filtered = HIP.mean_filter(image, kernel, num_iterations=1)
        print(f"[OK] Mean filter: input {image.shape} -> output {mean_filtered.shape}")

        # Test identity filter (simple passthrough)
        print("Testing identity filter...")
        identity = HIP.identity_filter(image)
        print(f"[OK] Identity filter: input {image.shape} -> output {identity.shape}")

        return True
    except Exception as e:
        print(f"[FAILED] Image processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print()


def test_help_system():
    """Test the help system."""
    print("=" * 60)
    print("Test 5: Help System")
    print("=" * 60)
    try:
        # Test getting help for a specific command
        # Note: Help() prints directly to stdout and returns None
        print("Calling HIP.help('Gaussian')...")
        print("-" * 60)
        HIP.help('Gaussian')
        print("-" * 60)
        print("[OK] Help function executed successfully")
        print("  (Help text printed above)")

        return True
    except Exception as e:
        print(f"[FAILED] Help system test failed: {e}")
        return False
    finally:
        print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print(" " * 10 + "Hydra Image Processor Package Test")
    print("=" * 60)
    print()

    results = []

    # Run tests
    results.append(test_import())
    results.append(test_device_info())
    results.append(test_utility_functions())
    results.append(test_image_processing())
    results.append(test_help_system())

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} tests")

    if all(results):
        print("\n[OK] All tests passed successfully!")
        return 0
    else:
        print("\n[FAILED] Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

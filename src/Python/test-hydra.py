#!/usr/bin/env python3
"""
Test script for the Hydra Image Processor Python module.

This script tests:
1. Module import
2. Basic CUDA functionality
3. Simple image processing operations
"""

import sys
import os
import numpy as np

# Add the directory containing the Hydra module to Python path
module_dir = os.path.dirname(os.path.abspath(__file__))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)


def test_import():
    """Test that the Hydra module can be imported."""
    try:
        import Hydra
        print("[SUCCESS] Successfully imported Hydra module")
        return True
    except ImportError as e:
        print(f"[FAILED] Failed to import Hydra: {e}")
        return False


def test_device_info():
    """Test CUDA device detection."""
    import Hydra

    try:
        # Get device count
        device_count = Hydra.DeviceCount()
        print(f"[SUCCESS] Found {device_count} CUDA device(s)")

        # Get device stats
        if device_count > 0:
            stats = Hydra.DeviceStats()
            print("  Device memory info available")

        return True
    except Exception as e:
        print(f"[WARNING] CUDA not available or error accessing devices: {e}")
        return False


def test_basic_operation():
    """Test a basic image processing operation."""
    import Hydra

    try:
        # Create a simple test image
        test_image = np.random.rand(100, 100).astype(np.float32)

        # Try a simple operation (e.g., Gaussian filter)
        # Parameters may vary based on your actual API
        result = Hydra.Gaussian(test_image, sigma=[2.0, 2.0])

        print("[SUCCESS] Successfully performed Gaussian filtering")
        print(f"  Input shape: {test_image.shape}, Output shape: {result.shape}")

        return True
    except Exception as e:
        print(f"[WARNING] Could not perform image processing: {e}")
        return False


def test_info():
    """Display available Hydra commands."""
    import Hydra

    try:
        info = Hydra.Info()
        print(f"[SUCCESS] Hydra module has {len(info)} available commands")

        # Print first few commands as examples
        print("  Sample commands:")
        for cmd in info[:5]:
            if hasattr(cmd, 'command'):
                print(f"    - {cmd.command}")

        return True
    except Exception as e:
        print(f"[WARNING] Could not get module info: {e}")
        return False


def main():
    """
    Run all tests.
    """
    print("=" * 60)
    print("Hydra Image Processor Module Test")
    print("=" * 60)

    # Track test results
    results = []

    # Test 1: Import
    print("\n1. Testing module import...")
    if not test_import():
        print("\nModule import failed. Cannot continue with other tests.")
        return 1
    results.append(True)

    # Test 2: Device info
    print("\n2. Testing CUDA device detection...")
    results.append(test_device_info())

    # Test 3: Module info
    print("\n3. Testing module information...")
    results.append(test_info())

    # Test 4: Basic operation
    print("\n4. Testing basic image processing...")
    results.append(test_basic_operation())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Passed: {sum(results)}/{len(results)} tests")

    if all(results):
        print("\n[SUCCESS] All tests passed successfully!")
    else:
        print("\n[WARNING] Some tests failed or had warnings.")
        print("  This may be expected if CUDA is not available on this system.")

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())

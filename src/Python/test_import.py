"""
Test script to verify Hydra module imports correctly with bundled OpenMP DLL.

This demonstrates that the module is portable and can be distributed
with just the Hydra.pyd and libomp140.x86_64.dll files together.
"""

import os
import sys


def test_hydra_import():
    """Test that Hydra imports successfully."""
    print("Testing Hydra import...")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")

    # Check required files exist
    required_files = ['Hydra.pyd', 'libomp140.x86_64.dll']
    print("\nChecking required files:")
    for filename in required_files:
        exists = os.path.exists(filename)
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {filename}: {exists}")

    # Import Hydra
    print("\nImporting Hydra module...")
    try:
        import Hydra
        print("[OK] Successfully imported Hydra!")
        print(f"  Module location: {Hydra.__file__}")

        # List available functions
        funcs = [name for name in dir(Hydra) if not name.startswith('_')]
        print(f"\n  Available functions ({len(funcs)} total):")
        for func in funcs[:5]:
            print(f"    - {func}")
        if len(funcs) > 5:
            print(f"    ... and {len(funcs) - 5} more")

        return True
    except ImportError as e:
        print(f"[FAILED] Failed to import Hydra: {e}")
        return False


if __name__ == "__main__":
    success = test_hydra_import()
    sys.exit(0 if success else 1)

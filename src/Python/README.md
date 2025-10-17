# Hydra Python Module

This directory contains the Hydra Python extension module for GPU-accelerated image processing.

## Files

- **Hydra.pyd** - The Python extension module (Windows)
- **libomp140.x86_64.dll** - OpenMP runtime library (bundled for portability)
- **test_import.py** - Test script to verify the module imports correctly

## Distribution

To distribute this module to others, simply copy both files:
1. `Hydra.pyd`
2. `libomp140.x86_64.dll`

Keep them in the same directory, and Python will automatically find the DLL when importing the module.

## Requirements

- Python 3.12
- NumPy
- NVIDIA CUDA-capable GPU (for running the image processing operations)

## Usage

```python
import Hydra

# Check available CUDA devices
num_devices = Hydra.DeviceCount()
print(f"Available CUDA devices: {num_devices}")

# Get help on available functions
Hydra.Help()

# Example: Apply Gaussian filter
# result = Hydra.Gaussian(input_array, sigma=1.5)
```

## Testing

Run the test script to verify the module is working:

```bash
python test_import.py
```

## Building from Source

The module is built using CMake with the following key features:
- Linked against Python 3.12
- Uses LLVM OpenMP (`/openmp:llvm`) for better performance
- Statically links CUDA runtime
- Automatically bundles the OpenMP DLL for portability

To rebuild:

```bash
cd ../..  # Go to project root
cmake --preset dev
cmake --build build --config Release --target HydraPy
```

The build system will automatically:
1. Find the correct Python 3.12 installation
2. Link against the hydra conda environment
3. Copy the OpenMP DLL to this directory

## Notes

- The module requires the OpenMP DLL at runtime. This is automatically bundled during the build process.
- The module was compiled against Python 3.12.11 and requires that specific Python version.
- CUDA runtime is statically linked, so no separate CUDA runtime installation is needed for basic import.
- However, NVIDIA GPU drivers must be installed to actually run GPU operations.

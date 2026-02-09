# Hydra Image Processor

## Project Overview

Hydra Image Processor (Hydra) is a high-performance library for signal filters and image analysis, capable of handling 1-5 dimensional data (x, y, z, channels, time). Its core feature is the ability to efficiently process data larger than GPU memory by optimally chunking it and distributing higher-dimensional chunks across multiple GPUs, while ensuring energy-insulated boundary conditions to prevent edge artifacts.

The project consists of a core C++/CUDA library with bindings for:
*   **Python**: A Conda-installable package `hydra-image-processor`.
*   **MATLAB**: A toolbox interface.

## Key Technologies

*   **Languages**: C++, CUDA, Python, MATLAB.
*   **Build System**: CMake, Ninja.
*   **Package Managers**: Conda, Pip.
*   **CI/CD**: GitHub Actions.

## Directory Structure

*   `src/c/Cuda`: Core C++/CUDA implementation of image processing kernels and chunking logic.
*   `src/c/Python`: C++ source for the Python extension module (`Hydra.pyd`/`Hydra.so`).
*   `src/c/Mex`: C++ source for the MATLAB MEX interface.
*   `src/Python`: Source code for the Python package (`hydra_image_processor`).
*   `src/MATLAB`: MATLAB toolbox code and wrappers.
*   `recipe/`: Conda build recipe (`meta.yaml`, `build.sh`, `bld.bat`).
*   `.github/workflows`: CI/CD workflows for building and publishing.

## Building and Running

### Prerequisites

*   NVIDIA GPU with CUDA support.
*   CUDA Toolkit (11.x or newer recommended).
*   CMake (3.22+).
*   C++ Compiler (MSVC on Windows, GCC/Clang on Linux).
*   Python 3.9+.

### Python Build (Local Development)

To build and install the Python package locally:

```bash
# Navigate to the project root
cd E:\programming\hydra-image-processor

# Install in editable mode (requires scikit-build-core or manual cmake invocation usually, 
# but this project uses a standard setup.py/cmake flow or pip install .)
pip install .
```

Alternatively, using the provided CMake presets:

```bash
cmake --preset dev
cmake --build build --config Release --target HydraPy
```

### Conda Package Build

To build the Conda package:

```bash
conda install conda-build
conda build recipe
```

### MATLAB Build

The MATLAB toolbox is built using the `.prj` file:
`src/MATLAB/HydraImageProcessor.prj`.

## Development Conventions

*   **Hybrid Implementation**: The Python package uses a "GPU-first, CPU-fallback" pattern. Wrappers in `src/Python/hydra_image_processor/core.py` try to call the compiled CUDA module and catch exceptions to fall back to local CPU implementations.
*   **Chunking Strategy**: Large images are automatically split into chunks that fit in GPU memory. This logic is handled in the C++ core (`ImageChunk.cpp`).
*   **Packaging**: The Python package includes the compiled extension binary (`.pyd` or `.so`) and required DLLs (like `libomp`) to ensure portability.

cmake_minimum_required(VERSION 3.23)

# Default GPU architectures when the caller doesn't specify any.
# 'all-major' (requires CMake >= 3.23) targets all major real architectures
# supported by the toolkit, plus PTX of the newest for forward compatibility.
# conda-forge overrides this via CMAKE_ARGS; developers can override with
# -DCMAKE_CUDA_ARCHITECTURES=... (e.g. "native" for a single local GPU).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "all-major" CACHE STRING "CUDA architectures to build for")
endif()

project(HydraImageProcessor LANGUAGES C CXX CUDA)

set(HYDRA_MODULE_NAME "Hydra")

# Tests build by default for local development; wheel/feedstock builds pass
# -DHYDRA_BUILD_TESTS=OFF to skip the standalone C++ accuracy executable.
option(HYDRA_BUILD_TESTS "Build the C++ accuracy tests" ON)

# Use CMake's modern FindCUDAToolkit module
find_package(CUDAToolkit REQUIRED)

# Find OpenMP - uses llvm-openmp from conda-forge on Windows
find_package(OpenMP)

# Find optional dependencies
find_package(Matlab COMPONENTS MAIN_PROGRAM OPTIONAL_COMPONENTS MEX_COMPILER)
find_package(Python3 COMPONENTS Development NumPy OPTIONAL)


# Setup backend Hydra CUDA library (static) for CUDA building
add_subdirectory(src/c/Cuda)

if (HYDRA_BUILD_TESTS)
    add_subdirectory(src/c/test_back)
endif()

# Setup MEX interface if Matlab was found
if (Matlab_FOUND)
    add_subdirectory(src/c/Mex)
endif()

# Setup Python interface if Python is found
if (Python3_FOUND)
    add_subdirectory(src/c/Python)
endif()

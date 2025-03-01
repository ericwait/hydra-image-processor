cmake_minimum_required(VERSION 3.22)

project(HydraImageProcessor LANGUAGES C CXX CUDA)

set(HYDRA_MODULE_NAME "HIP")

# Use CMake's modern FindCUDAToolkit module
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

# Find optional dependencies
find_package(Matlab COMPONENTS MAIN_PROGRAM OPTIONAL_COMPONENTS MEX_COMPILER)
find_package(Python COMPONENTS Development NumPy OPTIONAL)


# Setup backend Hydra CUDA library (static) for CUDA building
add_subdirectory(src/c/Cuda)
add_subdirectory(src/c/test_back)

# Setup MEX interface if Matlab was found
if (Matlab_FOUND)
    add_subdirectory(src/c/Mex)
endif()

# Setup Python interface if Python is found
if (Python_FOUND)
    add_subdirectory(src/c/Python)
endif()

cmake_minimum_required(VERSION 3.22)

project(HydraImageProcessor LANGUAGES C CXX CUDA)

set(HYDRA_MODULE_NAME "Hydra")

# Use CMake's modern FindCUDAToolkit module
find_package(CUDAToolkit REQUIRED)

# Find OpenMP - prefer LLVM implementation on MSVC for static linking
if(MSVC)
    # Set flags before finding OpenMP to prefer LLVM implementation
    set(OpenMP_C_FLAGS "/openmp:llvm" CACHE STRING "" FORCE)
    set(OpenMP_CXX_FLAGS "/openmp:llvm" CACHE STRING "" FORCE)
    set(OpenMP_C_LIB_NAMES "libomp" CACHE STRING "" FORCE)
    set(OpenMP_CXX_LIB_NAMES "libomp" CACHE STRING "" FORCE)
    set(OpenMP_libomp_LIBRARY "libomp" CACHE STRING "" FORCE)
endif()
find_package(OpenMP)

# Find optional dependencies
find_package(Matlab COMPONENTS MAIN_PROGRAM OPTIONAL_COMPONENTS MEX_COMPILER)
find_package(Python3 COMPONENTS Development NumPy OPTIONAL)


# Setup backend Hydra CUDA library (static) for CUDA building
add_subdirectory(src/c/Cuda)
add_subdirectory(src/c/test_back)

# Setup MEX interface if Matlab was found
if (Matlab_FOUND)
    add_subdirectory(src/c/Mex)
endif()

# Setup Python interface if Python is found
if (Python3_FOUND)
    add_subdirectory(src/c/Python)
endif()

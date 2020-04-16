cmake_minimum_required(VERSION 3.15.0)
cmake_policy(VERSION 3.15)

project(HydraImageProcessor LANGUAGES C CXX CUDA)

set(HYDRA_MODULE_NAME "HIP")

# NOTE: Use CUDACXX (gcc/clang) or -T cuda=<cuda_toolkit> (vs)
# TODO: Try to make selection simpler with cmake 3.17's FindCUDAToolkit?
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

find_package(Matlab COMPONENTS MAIN_PROGRAM)
find_package(Python COMPONENTS Development NumPy)

# Setup backend Hydra CUDA library (static) for cuda building
add_subdirectory(src/c/Cuda)

# Setup MEX interface if Matlab was found
if ( Matlab_FOUND )
    add_subdirectory(src/c/Mex)
endif()

# Setup Python interface if Python3 found
if ( Python_FOUND )
    add_subdirectory(src/c/Python)
endif()

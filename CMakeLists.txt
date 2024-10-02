cmake_minimum_required(VERSION 3.22)

project(HydraImageProcessor LANGUAGES C CXX CUDA)

set(HYDRA_MODULE_NAME "HIP")

# Use CMake's modern FindCUDAToolkit module
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Find optional dependencies
find_package(Matlab COMPONENTS MAIN_PROGRAM OPTIONAL_COMPONENTS MexCompiler)
find_package(Python COMPONENTS Development NumPy OPTIONAL)

# Set OpenMP and CUDA flags depending on the platform
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # For MSVC (Windows)
    set(OPENMP_FLAG "/openmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/openmp")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # For GCC/Clang (Linux/macOS)
    set(OPENMP_FLAG "-fopenmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-fopenmp")
endif()

# Apply the flags globally, or apply to specific targets later
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAG}")

# Setup backend Hydra CUDA library (static) for CUDA building
add_subdirectory(src/c/Cuda)

# Setup MEX interface if Matlab was found
if (Matlab_FOUND)
    add_subdirectory(src/c/Mex)
endif()

# Setup Python interface if Python is found
if (Python_FOUND)
    add_subdirectory(src/c/Python)
endif()

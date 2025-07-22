cmake_minimum_required(VERSION 3.22)

# Only use these if not set externally
if(NOT DEFINED GITVERSION_SEMVER)
  set(GITVERSION_SEMVER "0.0.0-dev")
endif()
if(NOT DEFINED GITVERSION_MAJOR)
  set(GITVERSION_MAJOR 0)
endif()
if(NOT DEFINED GITVERSION_MINOR)
  set(GITVERSION_MINOR 0)
endif()
if(NOT DEFINED GITVERSION_PATCH)
  set(GITVERSION_PATCH 0)
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/c/Version.h.in
  ${CMAKE_BINARY_DIR}/generated/Version.h
  @ONLY
)

include_directories(${CMAKE_BINARY_DIR}/generated)

# ===========================
# CUDA Architecture Targets
# ===========================
# Uses modern format (native for CUDA ≥ 11.1)
# Allows user override from command line: -DCMAKE_CUDA_ARCHITECTURES=...
# Use all known GPU archs by default, but allow user to override
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89;90;103;121"
        CACHE STRING "CUDA architectures to build for"
        FORCE)
endif()

project(HydraImageProcessor LANGUAGES C CXX CUDA)

set(HYDRA_MODULE_NAME "HIP")

# Use CMake's modern FindCUDAToolkit module
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

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

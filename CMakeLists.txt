cmake_minimum_required(VERSION 3.22)

# Project metadata
project(HydraImageProcessor VERSION 0.0.0 LANGUAGES C CXX CUDA)

# ===========================
# GitVersioning (optional defaults for local builds)
# ===========================
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

# ===========================
# Version Header Generation
# ===========================
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/c/Version.h.in
  ${CMAKE_BINARY_DIR}/generated/Version.h
  @ONLY
)
include_directories(${CMAKE_BINARY_DIR}/generated)

# ===========================
# CUDA Architecture Defaults
# ===========================
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "89;90;103;121"
      CACHE STRING "CUDA architectures to build for"
      FORCE)
endif()

# ===========================
# Default Install Locations
# ===========================
include(GNUInstallDirs)

# ===========================
# Dependencies
# ===========================
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

find_package(Matlab COMPONENTS MAIN_PROGRAM OPTIONAL_COMPONENTS MEX_COMPILER)
# find_package(Python COMPONENTS Development NumPy OPTIONAL)

# ===========================
# Subdirectories
# ===========================
add_subdirectory(src/c/Cuda)
add_subdirectory(src/c/test_back)

if (Matlab_FOUND)
  add_subdirectory(src/c/Mex)
endif()

# Try to help FindPython by injecting the environment path (for local mamba or CI conda)
message(STATUS "CONDA_PREFIX: $ENV{CONDA_PREFIX}")
if(DEFINED ENV{CONDA_PREFIX})
  if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/python.exe" CACHE FILEPATH "Python interpreter")
  else()
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/python" CACHE FILEPATH "Python interpreter")
  endif()
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  message(STATUS "Python: ${Python_EXECUTABLE}")
  message(STATUS "Python Include Dir: ${Python_INCLUDE_DIRS}")
else()
  message(STATUS "Using system Python interpreter")
endif()

find_package(Python COMPONENTS Interpreter Development NumPy OPTIONAL)
# Manually set the Python library location if using conda/mamba on Windows
# if(WIN32 AND NOT Python_LIBRARIES)
#   set(Python_LIBRARIES "$ENV{CONDA_PREFIX}/libs/python311.lib")
# endif()

message(STATUS "Python: ${Python_EXECUTABLE}")
message(STATUS "Python Include Dir: ${Python_INCLUDE_DIRS}")
message(STATUS "Python Library: ${Python_LIBRARIES}")
message(STATUS "NumPy Include Dir: ${Python_NumPy_INCLUDE_DIRS}")

if (Python_FOUND)
  add_subdirectory(src/c/Python)
endif()

# ===========================
# CMake Package Config
# ===========================
include(CMakePackageConfigHelpers)

# Generate hydra-config.cmake
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hydra-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/hydra-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Hydra
)

# Generate hydra-config-version.cmake
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/hydra-config-version.cmake
  VERSION ${GITVERSION_SEMVER}
  COMPATIBILITY AnyNewerVersion
)

# Install config files
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/hydra-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/hydra-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Hydra
)

# Export all targets (defined in subdirs like Cuda)
install(EXPORT HydraTargets
  FILE HydraTargets.cmake
  NAMESPACE Hydra::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Hydra
)

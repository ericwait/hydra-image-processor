cmake_minimum_required(VERSION 3.22)  # Ensure good CUDA support

# ===========================
# Options
# ===========================
option(USE_PROCESS_MUTEX "Use process-level mutex to guard GPU calls" OFF)

project(HydraCuda LANGUAGES CXX CUDA)

# ===========================
# Common Includes and Sources
# ===========================
set(COMMON_INCLUDES
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(COMMON_HEADERS
    BufferConversions.h
    CHelpers.h
    CommandLineProgress.h
    CudaDeviceInfo.h
    CudaDeviceStats.h
    CudaUtilities.h
    CWrapperAutogen.h
    CWrappers.h
    Defines.h
    ImageChunk.h
    ImageView.h
    KernelGenerators.h
    Vec.h
)

set(COMMON_CUDA_SOURCES
    CudaAddTwoImages.cuh
    CudaClosure.cuh
    CudaDeviceCount.cuh
    CudaDeviceImages.cuh
    CudaElementWiseDifference.cuh
    CudaEntropyFilter.cuh
    CudaGaussian.cuh
    CudaGetMinMax.cuh
    CudaHighPassFilter.cuh
    CudaIdentityFilter.cuh
    CudaImageContainer.cuh
    CudaImageContainerClean.cuh
    CudaLoG.cuh
    CudaMaxFilter.cuh
    CudaMeanAndVariance.cuh
    CudaMeanFilter.cuh
    CudaMedianFilter.cuh
    CudaMemoryStats.cuh
    CudaMinFilter.cuh
    CudaMinMax.cuh
    CudaMultiplySum.cuh
    CudaNLMeans.cuh
    CudaOpener.cuh
    CudaStdFilter.cuh
    CudaSum.cuh
    CudaVarFilter.cuh
    CudaWienerFilter.cuh
    ImageDimensions.cuh
    Kernel.cuh
    KernelIterator.cuh
    SeparableMultiplySum.cuh
)

set(COMMON_CPP_SOURCES
    CHelpers.cpp
    CudaDeviceInfo.cpp
    CudaDeviceStats.cpp
    CudaUtilities.cpp
    GaussianKernel.cpp
    ImageChunk.cpp
    LoGKernel.cpp

    CWrapperAutogen.cu
    CWrappers.cu
    Kernel.cu
    KernelIterator.cu
)

# ===========================
# OpenMP Setup
# ===========================
find_package(OpenMP REQUIRED COMPONENTS CXX)

# ===========================
# CUDA Library Builder Function
# ===========================
function(setup_cuda_library LIB_NAME LIB_TYPE)
    add_library(${LIB_NAME} ${LIB_TYPE})

    target_compile_features(${LIB_NAME} PRIVATE cxx_std_17)

    target_include_directories(${LIB_NAME}
        PUBLIC ${COMMON_INCLUDES}
    )

    target_sources(${LIB_NAME}
        PUBLIC ${COMMON_HEADERS}
        PRIVATE ${COMMON_CUDA_SOURCES} ${COMMON_CPP_SOURCES}
    )

    # Link CUDA and OpenMP
    target_link_libraries(${LIB_NAME}
        PRIVATE CUDA::cudart_static
        PRIVATE OpenMP::OpenMP_CXX
    )

    # CUDA-specific compile options
    target_compile_options(${LIB_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
    )

    set_target_properties(${LIB_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
endfunction()

# ===========================
# Library Targets
# ===========================
setup_cuda_library(HydraCudaStatic STATIC)
# setup_cuda_library(HydraCuda SHARED)
# Set a variable to turn on/off PROCESS_MUTEX support
option(USE_PROCESS_MUTEX "Use process-level mutex to guard GPU calls" OFF)

# Common settings for both libraries
set(CUDA_ARCHITECTURES "52;60;61;70")
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

# Function to set up both static and shared CUDA libraries
function(setup_cuda_library LIB_NAME STATIC_OR_SHARED)
    add_library(${LIB_NAME} ${STATIC_OR_SHARED} "")
    target_compile_features(${LIB_NAME} PRIVATE cxx_std_17)
    target_include_directories(${LIB_NAME}
        PUBLIC ${COMMON_INCLUDES}
    )
    target_link_libraries(${LIB_NAME} PRIVATE OpenMP::OpenMP_CXX)

    # Set CUDA-specific compile options
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)

    set_target_properties(${LIB_NAME} PROPERTIES
            CUDA_STANDARD 11
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
            POSITION_INDEPENDENT_CODE ON
    )
    target_sources(${LIB_NAME}
        PUBLIC ${COMMON_HEADERS}
        PRIVATE ${COMMON_CUDA_SOURCES} ${COMMON_CPP_SOURCES}
    )
endfunction()

# Setup static and shared libraries
setup_cuda_library(HydraCudaStatic STATIC)
setup_cuda_library(HydraCuda SHARED)

# Setup backend Hydra CUDA library (static) for cuda building
add_library(HydraCudaStatic STATIC "")

target_compile_features(HydraCudaStatic PRIVATE cxx_std_11)
target_link_libraries(HydraCudaStatic PRIVATE OpenMP::OpenMP_CXX)

# Set a variable to turn on/off PROCESS_MUTEX support
set(USE_PROCESS_MUTEX ON CACHE BOOL "Use process-level mutex to guard GPU calls")

# Complie compute architectures
# NOTE: Comment out some of these to speed up builds if you know your hardware capabilities
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_30,code=sm_30")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_35,code=sm_35")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_60")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=compute_70")

set_target_properties(HydraCudaStatic
    PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 11
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "52;60;61;70"
)

target_sources(HydraCudaStatic
    PUBLIC
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

    PRIVATE
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

target_include_directories(HydraCudaStatic 
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Backend shared lib for dynamic c++ linking
add_library(HydraCuda SHARED "")

target_compile_features(HydraCuda PRIVATE cxx_std_11)
target_link_libraries(HydraCuda 
    PRIVATE
        OpenMP::OpenMP_CXX
)

# Set a variable to turn on/off PROCESS_MUTEX support
# set(USE_PROCESS_MUTEX ON CACHE BOOL "Use process-level mutex to guard GPU calls")
set_target_properties(HydraCuda 
    PROPERTIES
        CUDA_STANDARD 11
        CUDA_ARCHITECTURES "52;60;61;70"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        LIBRARY_OUTPUT_DIRECTORY $<INSTALL_INTERFACE:lib>
)

target_sources(HydraCuda
    PUBLIC
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
    PRIVATE
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

target_include_directories(HydraCuda
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

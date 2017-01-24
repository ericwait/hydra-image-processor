#ifndef CUDA_RESIZE_CUH
#define CUDA_RESIZE_CUH

#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"
#include "CudaMedianFilter.cuh"

#include "Vec.h"
#include <vector>
#include "ImageChunk.cuh"
#include "CudaImageContainerClean.cuh"
#include "Defines.h"

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

template <class PixelType>
PixelType* cResize(const PixelType* imageIn, Vec<size_t> dims, Vec<double> resizeFactors, Vec<size_t>& resizeDims,
                   ReductionMethods method = REDUC_MEAN, PixelType** imageOut = NULL, int device = 0)
{
    cudaSetDevice(device);
    resizeDims = Vec<size_t>(Vec<double>(dims)/resizeFactors);

    PixelType* resizedImage = NULL;
    if(imageOut==NULL)
        resizedImage = new PixelType[resizeDims.product()];
    else
        resizedImage = *imageOut;

    if(resizeFactors.product()==1)
    {
        memcpy(imageOut, imageIn, sizeof(PixelType)*dims.product());
        return imageOut;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    size_t memAvail, total;
    cudaMemGetInfo(&memAvail, &total);

    std::vector<ImageChunk> orgChunks;
    std::vector<ImageChunk> resizedChunks;

    return resizedImage;
}

#endif

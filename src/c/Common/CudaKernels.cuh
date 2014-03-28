#pragma once

#define DEVICE_VEC
#include "Vec.h"
#include "Defines.h"
#include "cuda_runtime.h"
#include "CudaImageContainer.cuh"

__constant__ extern float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];

__global__ void cudaAddFactor(CudaImageContainer imageIn1, CudaImageContainer imageOut, double factor, DevicePixelType minValue,
							  DevicePixelType maxValue);

__global__ void cudaAddTwoImagesWithFactor(CudaImageContainer imageIn1, CudaImageContainer imageIn2, CudaImageContainer imageOut,
										   double factor, DevicePixelType minValue, DevicePixelType maxValue);

__device__ DevicePixelType cudaFindMedian(DevicePixelType* vals, int numVals);

__global__ void cudaFindMinMax(CudaImageContainer imageIn, double* minArrayOut, double* maxArrayOut, size_t n);

__global__ void cudaGetROI(CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostStartPos, Vec<size_t> hostNewSize);

__global__ void cudaHistogramCreate( CudaImageContainer imageIn, size_t* histogram);

__global__ void cudaMask(const CudaImageContainer imageIn1, const CudaImageContainer imageIn2, CudaImageContainer imageOut,
						 DevicePixelType threshold);

__global__ void cudaMaxFilter( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims, DevicePixelType  minVal,
							  DevicePixelType maxVal);

__global__ void cudaMaximumIntensityProjection(CudaImageContainer imageIn, CudaImageContainer imageOut);

__global__ void cudaMeanFilter(CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims);

__global__ void cudaMeanImageReduction(CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostReductions);

__global__ void cudaMedianFilter(CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims);

__global__ void cudaMedianImageReduction(CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostReductions);

__global__ void cudaMinFilter( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims, DevicePixelType  minVal,
							  DevicePixelType maxVal);

__global__ void cudaMultAddFilter(CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims, size_t kernelOffset=0);

__global__ void cudaMultiplyImage(CudaImageContainer imageIn, CudaImageContainer imageOut, double factor, DevicePixelType minValue,
								  DevicePixelType maxValue);

__global__ void cudaMultiplyTwoImages(CudaImageContainer imageIn1, CudaImageContainer imageIn2, CudaImageContainer imageOut,
									  double factor, DevicePixelType minValue, DevicePixelType maxValue);

__global__ void cudaNormalizeHistogram(size_t* histogram, double* normHistogram, Vec<size_t> imageDims);

__global__ void cudaPolyTransferFuncImage(CudaImageContainer imageIn, CudaImageContainer imageOut, double a, double b, double c,
										  DevicePixelType minPixelValue, DevicePixelType maxPixelValue);

__global__ void cudaPow(CudaImageContainer imageIn1, CudaImageContainer imageOut, double power, DevicePixelType minValue,
						DevicePixelType maxValue);

__global__ void cudaSumArray(CudaImageContainer arrayIn, double* arrayOut, size_t n);

__global__ void cudaThresholdImage(CudaImageContainer imageIn, CudaImageContainer imageOut, DevicePixelType threshold,
								   DevicePixelType minValue, DevicePixelType maxValue);

__global__ void cudaUnmixing(const CudaImageContainer imageIn1, const CudaImageContainer imageIn2, CudaImageContainer imageOut1,
							 Vec<size_t> hostKernelDims, DevicePixelType minPixelValue, DevicePixelType maxPixelValue);

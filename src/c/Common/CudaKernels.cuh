#pragma once

#define DEVICE_VEC
#include "Vec.h"
#include "Defines.h"
#include "cuda_runtime.h"
#include "CudaImageContainer.cuh"

__global__ void cudaAddFactor(CudaImageContainer<DevicePixelType> imageIn1, CudaImageContainer<DevicePixelType> imageOut, double factor, DevicePixelType minValue,
							  DevicePixelType maxValue);

__global__ void cudaAddTwoImagesWithFactor(CudaImageContainer<DevicePixelType> imageIn1, CudaImageContainer<DevicePixelType> imageIn2, CudaImageContainer<DevicePixelType> imageOut,
										   double factor, DevicePixelType minValue, DevicePixelType maxValue);

__device__ DevicePixelType cudaFindMedian(DevicePixelType* vals, int numVals);

__global__ void cudaFindMinMax(CudaImageContainer<DevicePixelType> imageIn, double* minArrayOut, double* maxArrayOut, size_t n);

__global__ void cudaGetROI(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, Vec<size_t> hostStartPos, Vec<size_t> hostNewSize);

__global__ void cudaHistogramCreate( CudaImageContainer<DevicePixelType> imageIn, size_t* histogram);

__global__ void cudaMask(const CudaImageContainer<DevicePixelType> imageIn1, const CudaImageContainer<DevicePixelType> imageIn2, CudaImageContainer<DevicePixelType> imageOut,
						 DevicePixelType threshold);

__global__ void cudaMaxFilter( CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, Vec<size_t> hostKernelDims, DevicePixelType minVal,
							  DevicePixelType maxVal);

__global__ void cudaMaximumIntensityProjection(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut);

__global__ void cudaMeanFilter(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, Vec<size_t> hostKernelDims);

__global__ void cudaMeanImageReduction(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, Vec<size_t> hostReductions);

__global__ void cudaMedianFilter(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, Vec<size_t> hostKernelDims);

__global__ void cudaMedianImageReduction(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, Vec<size_t> hostReductions);

__global__ void cudaMinFilter( CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, Vec<size_t> hostKernelDims, DevicePixelType minVal,
							  DevicePixelType maxVal);

__global__ void cudaMultAddFilter(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, Vec<size_t> hostKernelDims, size_t kernelOffset=0);

__global__ void cudaMultiplyImage(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, double factor, DevicePixelType minValue,
								  DevicePixelType maxValue);

__global__ void cudaMultiplyTwoImages(CudaImageContainer<DevicePixelType> imageIn1, CudaImageContainer<DevicePixelType> imageIn2, CudaImageContainer<DevicePixelType> imageOut,
									  double factor, DevicePixelType minValue, DevicePixelType maxValue);

__global__ void cudaNormalizeHistogram(size_t* histogram, double* normHistogram, Vec<size_t> imageDims);

__global__ void cudaPolyTransferFuncImage(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, double a, double b, double c,
										  DevicePixelType minPixelValue, DevicePixelType maxPixelValue);

__global__ void cudaPow(CudaImageContainer<DevicePixelType> imageIn1, CudaImageContainer<DevicePixelType> imageOut, double power, DevicePixelType minValue,
						DevicePixelType maxValue);

__global__ void cudaSumArray(DevicePixelType* arrayIn, double* arrayOut, size_t n);

__global__ void cudaThresholdImage(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, DevicePixelType threshold,
								   DevicePixelType minValue, DevicePixelType maxValue);

__global__ void cudaUnmixing(const CudaImageContainer<DevicePixelType> imageIn1, const CudaImageContainer<DevicePixelType> imageIn2, CudaImageContainer<DevicePixelType> imageOut1,
							 Vec<size_t> hostKernelDims, DevicePixelType minPixelValue, DevicePixelType maxPixelValue);

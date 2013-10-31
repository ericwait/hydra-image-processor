#pragma once

#define DEVICE_VEC
#include "Vec.h"
#include "Defines.h"
#include "cuda_runtime.h"

__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];

template<typename ImagePixelType>
__global__ void cudaMeanFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageDims,
							   Vec<size_t> hostKernelDims)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		double val = 0;
		double kernelVolume = 0;
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageDims.z && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageDims.y && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageDims.x && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					val += imageIn[imageDims.linearAddressAt(curCoordIm)];
					++kernelVolume;
				}
			}
		}

		imageOut[imageDims.linearAddressAt(coordinate)] = val/kernelVolume;
	}
}

template<typename ImagePixelType>
__global__ void cudaMultiplyImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageDims, double factor,
	ImagePixelType minValue, ImagePixelType maxValue)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		imageOut[imageDims.linearAddressAt(coordinate)] = 
			min((double)maxValue,max((double)minValue, factor*imageIn[imageDims.linearAddressAt(coordinate)]));
	}
}

template<typename ImagePixelType1, typename ImagePixelType2, typename ImagePixelType3>//, typename FactorType>
__global__ void cudaAddTwoImagesWithFactor(ImagePixelType1* imageIn1, ImagePixelType2* imageIn2, ImagePixelType3* imageOut,
	Vec<size_t> hostImageDims, double factor, ImagePixelType3 minValue, ImagePixelType3 maxValue)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		double subtractor = factor*(double)imageIn2[imageDims.linearAddressAt(coordinate)];
		ImagePixelType3 outValue = (double)imageIn1[imageDims.linearAddressAt(coordinate)] + subtractor;

		imageOut[imageDims.linearAddressAt(coordinate)] = min(maxValue,max(minValue,outValue));
	}
}

template<typename ImagePixelType1, typename ImagePixelType2, typename ImagePixelType3>
__global__ void cudaMultiplyTwoImages(ImagePixelType1* imageIn1, ImagePixelType2* imageIn2, ImagePixelType3* imageOut,
	Vec<size_t> hostImageDims)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		size_t idx = imageDims.linearAddressAt(coordinate);
		ImagePixelType1 val1 = imageIn1[imageDims.linearAddressAt(coordinate)];
		ImagePixelType2 val2 = imageIn2[imageDims.linearAddressAt(coordinate)];
		imageOut[imageDims.linearAddressAt(coordinate)] = 
			imageIn1[imageDims.linearAddressAt(coordinate)] * imageIn2[imageDims.linearAddressAt(coordinate)];
	}
}

 template<typename ImagePixelType>
 __global__ void cudaAddFactor(ImagePixelType* imageIn1, ImagePixelType* imageOut, Vec<size_t> hostImageDims, double factor,
	 ImagePixelType minValue, ImagePixelType maxValue)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		double outValue = imageIn1[imageDims.linearAddressAt(coordinate)] + factor;
		imageOut[imageDims.linearAddressAt(coordinate)] = min((double)maxValue,max((double)minValue,outValue));
	}
}

 template <typename ImagePixelType>
 __device__ ImagePixelType* SubDivide(ImagePixelType* pB, ImagePixelType* pE)
 {
	 ImagePixelType* pPivot = --pE;
	 const ImagePixelType pivot = *pPivot;

	 while (pB < pE)
	 {
		 if (*pB > pivot)
		 {
			 --pE;
			 ImagePixelType temp = *pB;
			 *pB = *pE;
			 *pE = temp;
		 } else
			 ++pB;
	 }

	 ImagePixelType temp = *pPivot;
	 *pPivot = *pE;
	 *pE = temp;

	 return pE;
 }

 template <typename ImagePixelType>
 __device__ void SelectElement(ImagePixelType* pB, ImagePixelType* pE, size_t k)
 {
	 while (true)
	 {
		 ImagePixelType* pPivot = SubDivide(pB, pE);
		 size_t n = pPivot - pB;

		 if (n == k)
			 break;

		 if (n > k)
			 pE = pPivot;
		 else
		 {
			 pB = pPivot + 1;
			 k -= (n + 1);
		 }
	 }
 }

template<typename ImagePixelType>
__device__ ImagePixelType cudaFindMedian(ImagePixelType* vals, int numVals)
{
	SelectElement(vals,vals+numVals, numVals/2);
	return vals[numVals/2];
}

template<typename ImagePixelType>
__global__ void cudaMedianFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageDims, 
								 Vec<size_t> hostKernelDims)
{
	extern __shared__ ImagePixelType vals[];
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> kernelDims = hostKernelDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
	offset *=  kernelDims.product();

	if (coordinate<imageDims)
	{
		int kernelVolume = 0;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageDims.z && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageDims.y && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageDims.x && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					vals[kernelVolume+offset] = imageIn[imageDims.linearAddressAt(curCoordIm)];
					++kernelVolume;
				}
			}
		}

		imageOut[imageDims.linearAddressAt(coordinate)] = cudaFindMedian(vals+offset,kernelVolume);
		__syncthreads();
	}
}

template<typename ImagePixelType1, typename ImagePixelType2>
__global__ void cudaMultAddFilter(ImagePixelType1* imageIn, ImagePixelType2* imageOut, Vec<size_t> hostImageDims,
	Vec<size_t> hostKernelDims, int kernelOffset=0)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		double val = 0;
		double kernFactor = 0;

		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageDims.z && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageDims.y && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageDims.x && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					size_t kernIdx = kernelDims.linearAddressAt(curCoordKrn)+kernelOffset;
					kernFactor += cudaConstKernel[kernIdx];
					val += imageIn[imageDims.linearAddressAt(curCoordIm)] * cudaConstKernel[kernIdx];
				}
			}
		}

		imageOut[imageDims.linearAddressAt(coordinate)] = val/kernFactor;
	}
}

template<typename ImagePixelType>
__global__ void cudaMinFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageDims,
	Vec<size_t> hostKernelDims)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		ImagePixelType minVal = imageIn[imageDims.linearAddressAt(coordinate)];
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageDims.z && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageDims.y && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageDims.x && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					if(cudaConstKernel[kernelDims.linearAddressAt(curCoordKrn)]>0)
					{
						minVal = (ImagePixelType)min((float)minVal, imageIn[imageDims.linearAddressAt(curCoordIm)]*
							cudaConstKernel[kernelDims.linearAddressAt(curCoordKrn)]);
					}
				}
			}
		}

		imageOut[imageDims.linearAddressAt(coordinate)] = minVal;
	}
}

template<typename ImagePixelType>
__global__ void cudaMaxFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageDims,
	Vec<size_t> hostKernelDims)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		ImagePixelType maxVal = imageIn[imageDims.linearAddressAt(coordinate)];
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = (kernelDims.x+1)/2;
		kernelMidIdx.y = (kernelDims.y+1)/2;
		kernelMidIdx.z = (kernelDims.z+1)/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageDims.z && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageDims.y && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageDims.x && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					if(cudaConstKernel[kernelDims.linearAddressAt(curCoordKrn)]>0)
					{
						maxVal = (ImagePixelType)max((float)maxVal, imageIn[imageDims.linearAddressAt(curCoordIm)]*
							cudaConstKernel[kernelDims.linearAddressAt(curCoordKrn)]);
					}
				}
			}
		}

		imageOut[imageDims.linearAddressAt(coordinate)] = maxVal;
	}
}

template<typename ImagePixelType>
__global__ void cudaHistogramCreate(ImagePixelType* imageIn, size_t* histogram, Vec<size_t> hostImageDims)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	//This code is modified from that of Sanders - Cuda by Example
	__shared__ size_t tempHisto[NUM_BINS];
	tempHisto[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < imageDims.x*imageDims.y*imageDims.z)
	{
		atomicAdd(&(tempHisto[imageIn[i]]), 1);
		i += stride;
	}

	__syncthreads();
	atomicAdd(&(histogram[threadIdx.x]), tempHisto[threadIdx.x]);
}

template<typename ImagePixelType>
__global__ void cudaHistogramCreateROI(ImagePixelType* imageIn, size_t* histogram, Vec<size_t> starts,
	Vec<size_t> sizes)
{
// 	//This code is modified from that of Sanders - Cuda by Example
// 	__shared__ size_t tempHisto[NUM_BINS];
// 	tempHisto[threadIdx.x] = 0;
// 	__syncthreads();
// 
// 	int i = threadIdx.x + blockIdx.x * blockDim.x;
// 	int stride = blockDim.x * gridDim.x;
// 
// 	while (i < imageDims.x*imageDims.y*imageDims.z)
// 	{
// 		atomicAdd(&(tempHisto[imageIn[i]]), 1);
// 		i += stride;
// 	}
// 
// 	__syncthreads();
// 	atomicAdd(&(histogram[threadIdx.x]), tempHisto[threadIdx.x]);
}

__global__ void cudaNormalizeHistogram(size_t* histogram, double* normHistogram, Vec<size_t> imageDims)
{
	int x = blockIdx.x;
	normHistogram[x] = (double)(histogram[x]) / (imageDims.x*imageDims.y*imageDims.z);
}

template<typename ImagePixelType, typename ThresholdType>
__global__ void cudaThresholdImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageDims,
	ThresholdType threshold, ImagePixelType minValue, ImagePixelType maxValue)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		if (imageIn[imageDims.linearAddressAt(coordinate)]>=threshold)
			imageOut[imageDims.linearAddressAt(coordinate)] = maxValue;
		else
			imageOut[imageDims.linearAddressAt(coordinate)] = minValue;
	}
}

template<typename ImagePixelType>
__global__ void cudaFindMinMax(ImagePixelType* arrayIn, double* minArrayOut, double* maxArrayOut, size_t n)

{
	extern __shared__ double maxData[];
	extern __shared__ double minData[];

	size_t tid = threadIdx.x;
	size_t i = blockIdx.x*blockDim.x*2 + tid;
	size_t gridSize = blockDim.x*2*gridDim.x;

	while (i<n)
	{
		maxData[tid] = arrayIn[i];
		minData[tid] = arrayIn[i];

		if (i+blockDim.x<n)
		{
			if(maxData[tid]<arrayIn[i+blockDim.x])
				maxData[tid] = arrayIn[i+blockDim.x];

			if(minData[tid]>arrayIn[i+blockDim.x])
				minData[tid] = arrayIn[i+blockDim.x];
		}

		i += gridSize;
	}
	__syncthreads();


	if (blockDim.x >= 2048)
	{
		if (tid < 1024) 
		{
			if(maxData[tid]<maxData[tid + 1024])
				maxData[tid] = maxData[tid + 1024];

			if(minData[tid]>minData[tid + 1024])
				minData[tid] = minData[tid + 1024];
		}
		__syncthreads();
	}
	if (blockDim.x >= 1024)
	{
		if (tid < 512) 
		{
			if(maxData[tid]<maxData[tid + 512])
				maxData[tid] = maxData[tid + 512];

			if(minData[tid]>minData[tid + 512])
				minData[tid] = minData[tid + 512];
		}
		__syncthreads();
	}
	if (blockDim.x >= 512)
	{
		if (tid < 256) 
		{
			if(maxData[tid]<maxData[tid + 256])
				maxData[tid] = maxData[tid + 256];

			if(minData[tid]>minData[tid + 256])
				minData[tid] = minData[tid + 256];
		}
		__syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128)
		{
			if(maxData[tid]<maxData[tid + 128])
				maxData[tid] = maxData[tid + 128];

			if(minData[tid]>minData[tid + 128])
				minData[tid] = minData[tid + 128];
		}
		__syncthreads(); 
	}
	if (blockDim.x >= 128) 
	{
		if (tid < 64)
		{
			if(maxData[tid]<maxData[tid + 64])
				maxData[tid] = maxData[tid + 64];

			if(minData[tid]>minData[tid + 64])
				minData[tid] = minData[tid + 64];
		}
		__syncthreads(); 
	}

	if (tid < 32) {
		if (blockDim.x >= 64) 
		{
			{
				if(maxData[tid]<maxData[tid + 64])
					maxData[tid] = maxData[tid + 64];

				if(minData[tid]>minData[tid + 64])
					minData[tid] = minData[tid + 64];
			}
			__syncthreads(); 
		}
		if (blockDim.x >= 32)
		{
			if(maxData[tid]<maxData[tid + 16])
				maxData[tid] = maxData[tid + 16];

			if(minData[tid]>minData[tid + 16])
				minData[tid] = minData[tid + 16];
			__syncthreads(); 
		}
		if (blockDim.x >= 16)
		{
			if(maxData[tid]<maxData[tid + 8])
				maxData[tid] = maxData[tid + 8];

			if(minData[tid]>minData[tid + 8])
				minData[tid] = minData[tid + 8];
			__syncthreads(); 
		}
		if (blockDim.x >= 8)
		{
			if(maxData[tid]<maxData[tid + 4])
				maxData[tid] = maxData[tid + 4];

			if(minData[tid]>minData[tid + 4])
				minData[tid] = minData[tid + 4];
			__syncthreads(); 
		}
		if (blockDim.x >= 4)
		{
			if(maxData[tid]<maxData[tid + 2])
				maxData[tid] = maxData[tid + 2];

			if(minData[tid]>minData[tid + 2])
				minData[tid] = minData[tid + 2];
			__syncthreads(); 
		}
		if (blockDim.x >= 2)
		{
			if(maxData[tid]<maxData[tid + 1])
				maxData[tid] = maxData[tid + 1];

			if(minData[tid]>minData[tid + 1])
				minData[tid] = minData[tid + 1];
			__syncthreads(); 
		}
	}

	if (tid==0)
	{
		minArrayOut[blockIdx.x] = minData[0];
		maxArrayOut[blockIdx.x] = maxData[0];
	}
}

template<typename ImagePixelType>
__global__ void cudaPolyTransferFuncImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageDims,
	double a, double b, double c, ImagePixelType minPixelValue, ImagePixelType maxPixelValue)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		double pixVal = (double)imageIn[imageDims.linearAddressAt(coordinate)] / maxPixelValue;
		double multiplier = a*pixVal*pixVal + b*pixVal + c;
		if (multiplier<0)
			multiplier = 0;
		if (multiplier>1)
			multiplier = 1;

		ImagePixelType newPixelVal = min((double)maxPixelValue,max((double)minPixelValue, multiplier*maxPixelValue));

		imageOut[imageDims.linearAddressAt(coordinate)] = newPixelVal;
	}
}

template<typename DataType>
__global__ void cudaSumArray(DataType* arrayIn, double* arrayOut, size_t n)

{
	//This algorithm was used from a this website:
	// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
	// accessed 4/28/2013

	extern __shared__ double sdata[];

	size_t tid = threadIdx.x;
	size_t i = blockIdx.x*blockDim.x*2 + tid;
	size_t gridSize = blockDim.x*2*gridDim.x;
	sdata[tid] = 0;

	while (i<n)
	{
		sdata[tid] = arrayIn[i];

		if (i+blockDim.x<n)
			sdata[tid] += arrayIn[i+blockDim.x];

		i += gridSize;
	}
	__syncthreads();


	if (blockDim.x >= 2048)
	{
		if (tid < 1024) 
			sdata[tid] += sdata[tid + 1024];
		__syncthreads();
	}
	if (blockDim.x >= 1024)
	{
		if (tid < 512) 
			sdata[tid] += sdata[tid + 512];
		__syncthreads();
	}
	if (blockDim.x >= 512)
	{
		if (tid < 256) 
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads(); 
	}
	if (blockDim.x >= 128) 
	{
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads(); 
	}

	if (tid < 32) {
		if (blockDim.x >= 64) 
		{
			sdata[tid] += sdata[tid + 32];
			__syncthreads(); 
		}
		if (blockDim.x >= 32)
		{
			sdata[tid] += sdata[tid + 16];
			__syncthreads(); 
		}
		if (blockDim.x >= 16)
		{
			sdata[tid] += sdata[tid + 8];
			__syncthreads(); 
		}
		if (blockDim.x >= 8)
		{
			sdata[tid] += sdata[tid + 4];
			__syncthreads(); 
		}
		if (blockDim.x >= 4)
		{
			sdata[tid] += sdata[tid + 2];
			__syncthreads(); 
		}
		if (blockDim.x >= 2)
		{
			sdata[tid] += sdata[tid + 1];
			__syncthreads(); 
		}
	}

	if (tid==0)
		arrayOut[blockIdx.x] = sdata[0];
}

template<typename ImagePixelType>
__global__ void cudaRuduceImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageInDims,
	Vec<size_t> hostImageOutDims, Vec<double> hostReductions)
{
	DeviceVec<size_t> imageInDims = hostImageInDims;
	DeviceVec<size_t> imageOutDims = hostImageOutDims;
	DeviceVec<double> reductions = hostReductions;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageOutDims)
	{
		double val = 0;
		DeviceVec<size_t> mins, maxs;
		mins.x = coordinate.x*reductions.x;
		maxs.x = min(mins.x+reductions.x,(double)imageInDims.x);
		mins.y = coordinate.y*reductions.y;
		maxs.y = min(mins.y+reductions.y,(double)imageInDims.y);
		mins.z = coordinate.z*reductions.z;
		maxs.z = min(mins.z+reductions.z,(double)imageInDims.z);

		DeviceVec<size_t> currCorrd;
		for (currCorrd.z=mins.z; currCorrd.z<maxs.z; ++currCorrd.z)
		{
			for (currCorrd.y=mins.y; currCorrd.y<maxs.y; ++currCorrd.y)
			{
				for (currCorrd.x=mins.x; currCorrd.x<maxs.x; ++currCorrd.x)
				{
					val += (float)imageIn[imageInDims.linearAddressAt(currCorrd)];
				}
			}
		}

		imageOut[imageOutDims.linearAddressAt(coordinate)] = val/(maxs-mins).product();
	}
}

template<typename ImagePixelType>
__global__ void cudaMaximumIntensityProjection(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageDims,
											   bool isColumnMajor=false)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims && coordinate.z==0)
	{
		ImagePixelType maxVal = 0;
		for (; coordinate.z<imageDims.z; ++coordinate.z)
		{
			if (maxVal<imageIn[imageDims.linearAddressAt(coordinate,isColumnMajor)])
			{
				maxVal = imageIn[imageDims.linearAddressAt(coordinate,isColumnMajor)];
			}
		}

		coordinate.z = 0;
		imageOut[imageDims.linearAddressAt(coordinate,isColumnMajor)] = maxVal;
	}
}

template<typename ImagePixelType>
__global__ void cudaGetROI(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageDims,
	Vec<size_t> hostStartPos,	Vec<size_t> hostNewSize)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> newSize = hostNewSize;
	DeviceVec<size_t> startPos = hostStartPos;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate>=startPos && coordinate<startPos+newSize && coordinate<imageDims)
	{
		size_t outIndex = newSize.linearAddressAt(coordinate-startPos);
		imageOut[outIndex] = imageIn[imageDims.linearAddressAt(coordinate)];
	}
}

template<typename ImagePixelType, typename PowerType>
__global__ void cudaPow(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<size_t> hostImageDims, PowerType p,
						bool isColumnMajor=false)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
		imageOut[imageDims.linearAddressAt(coordinate,isColumnMajor)] = 
		pow((double)imageIn[imageDims.linearAddressAt(coordinate,isColumnMajor)],p);
}

template<typename ImagePixelType>
__global__ void cudaUnmixing(const ImagePixelType* imageIn1, const ImagePixelType* imageIn2, ImagePixelType* imageOut1,
	Vec<size_t> hostImageDims, Vec<size_t> hostKernelDims, ImagePixelType minPixelValue, ImagePixelType maxPixelValue,
	bool isColumnMajor=false)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		double meanIm1 = 0;
		double meanIm2 = 0;		int kernelVolume = 0;
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageDims.z && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageDims.y && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageDims.x && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					meanIm1 += imageIn1[imageDims.linearAddressAt(curCoordIm,isColumnMajor)];
					meanIm2 += imageIn2[imageDims.linearAddressAt(curCoordIm,isColumnMajor)];
					++kernelVolume;
				}
			}
		}

		meanIm1 /= kernelVolume;
		meanIm2 /= kernelVolume;

		if (meanIm1 < meanIm2)
		{
			imageOut1[imageDims.linearAddressAt(coordinate,isColumnMajor)] = 
				min(maxPixelValue,
				max(imageIn1[imageDims.linearAddressAt(coordinate,isColumnMajor)]-imageIn2[imageDims.linearAddressAt(coordinate,isColumnMajor)]
				,minPixelValue));
		}
		else 
		{
			imageOut1[imageDims.linearAddressAt(coordinate,isColumnMajor)] = imageIn1[imageDims.linearAddressAt(coordinate,isColumnMajor)];
		}	
	}
}

template<typename ImagePixelType>
__global__ void cudaMask(const ImagePixelType* imageIn1, const ImagePixelType* imageIn2, ImagePixelType* imageOut,
						 Vec<size_t> hostImageDims, ImagePixelType threshold)
{
	DeviceVec<size_t> imageDims = hostImageDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		ImagePixelType val=0;

		if (imageIn2[imageDims.linearAddressAt(coordinate)] <= threshold)
			val = imageIn1[imageDims.linearAddressAt(coordinate)];

		imageOut[imageDims.linearAddressAt(coordinate)] = val;
	}
}
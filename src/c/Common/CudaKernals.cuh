#pragma once

#define DEVICE_VEC
#include "Vec.h"

#include "Defines.h"
#include "CHelpers.h"
#include "cuda_runtime.h"

__constant__ float kernal[MAX_KERNAL_DIM*MAX_KERNAL_DIM*MAX_KERNAL_DIM];

template<typename ImagePixelType>
__global__ void cudaMeanFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims, Vec<int> hostKernalDims)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<int> kernalDims = hostKernalDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		double val = 0;
		int halfKwidth = kernalDims.x/2;
		int halfKheight = kernalDims.y/2;
		int halfKdepth = kernalDims.z/2;
		int iIm=-halfKwidth, iKer=0;
		DeviceVec<unsigned int> curCoord; 
		for (; iIm<halfKwidth; ++iIm, ++iKer)
		{
			curCoord.x = min(max(coordinate.x+iIm,0),imageDims.x-1);
			int jIm=-halfKheight, jKer=0;
			for (; jIm<halfKheight; ++jIm, ++jKer)
			{
				curCoord.y = min(max(coordinate.y+jIm,0),imageDims.y-1);
				int kIm=-halfKdepth, kKer=0;
				for (; kIm<halfKdepth; ++kIm, ++kKer)
				{
					curCoord.z = min(max(coordinate.z+kIm,0),imageDims.z-1);
					val += imageIn[imageDims.linearAddressAt(curCoord)];
				}
			}
		}

		imageOut[imageDims.linearAddressAt(coordinate)] = val/(kernalDims.x*kernalDims.y*kernalDims.z);
	}
}

template<typename ImagePixelType, typename FactorType>
__global__ void cudaMultiplyImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims, FactorType factor,
	ImagePixelType minValue, ImagePixelType maxValue)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		imageOut[imageDims.linearAddressAt(coordinate)] = 
			min(maxValue,max(minValue, factor*imageIn[imageDims.linearAddressAt(coordinate)]));
	}
}

template<typename ImagePixelType1, typename ImagePixelType2, typename ImagePixelType3>//, typename FactorType>
__global__ void cudaAddTwoImagesWithFactor(ImagePixelType1* imageIn1, ImagePixelType2* imageIn2, ImagePixelType3* imageOut,
	Vec<unsigned int> hostImageDims, double factor, ImagePixelType3 minValue, ImagePixelType3 maxValue)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		ImagePixelType3 outValue = 
			imageIn1[imageDims.linearAddressAt(coordinate)] + factor*imageIn2[imageDims.linearAddressAt(coordinate)];

		imageOut[imageDims.linearAddressAt(coordinate)] = min(maxValue,max(minValue,outValue));
			
	}
}

template<typename ImagePixelType1, typename ImagePixelType2, typename ImagePixelType3>
__global__ void cudaMultiplyTwoImages(ImagePixelType1* imageIn1, ImagePixelType2* imageIn2, ImagePixelType3* imageOut,
	Vec<unsigned int> hostImageDims)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		unsigned int idx = imageDims.linearAddressAt(coordinate);
		ImagePixelType1 val1 = imageIn1[imageDims.linearAddressAt(coordinate)];
		ImagePixelType2 val2 = imageIn2[imageDims.linearAddressAt(coordinate)];
		imageOut[imageDims.linearAddressAt(coordinate)] = 
			imageIn1[imageDims.linearAddressAt(coordinate)] * imageIn2[imageDims.linearAddressAt(coordinate)];
	}
}

 template<typename ImagePixelType, typename FactorType>
 __global__ void cudaAddFactor(ImagePixelType* imageIn1, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims, FactorType factor)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		imageOut[imageDims.linearAddressAt(coordinate)] = imageIn1[imageDims.linearAddressAt(coordinate)] + factor;
	}
}

template<typename ImagePixelType>
__device__ ImagePixelType cudaFindMedian(ImagePixelType* vals, int numVals)
{
	//TODO this algo could use some improvement
	int minIndex;
	ImagePixelType minValue;
	ImagePixelType tempValue;
	for (int i=0; i<=numVals/2; ++i)
	{
		minIndex = i;
		minValue = vals[i];
		for (int j=i+1; j<numVals; ++j)
		{
			if (vals[j]<minValue)
			{
				minIndex = j;
				minValue = vals[j];
			}
			tempValue = vals[i];
			vals[i] = vals[minIndex];
			vals[minIndex] = tempValue;
		}
	}
	//return vals[numVals/2];
	return vals[0];
}

template<typename ImagePixelType>
__global__ void cudaMedianFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims,
	Vec<int> hostKernalDims)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<int> kernalDims = hostKernalDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		ImagePixelType* vals = new ImagePixelType[kernalDims.x*kernalDims.y*kernalDims.z];
		int halfKwidth = kernalDims.x/2;
		int halfKheight = kernalDims.y/2;
		int halfKdepth = kernalDims.z/2;
		int iIm=-halfKwidth, iKer=0;
		DeviceVec<unsigned int> curCoord; 
		for (; iIm<halfKwidth; ++iIm, ++iKer)
		{
			curCoord.x = min(max(coordinate.x+iIm,0),imageDims.x-1);
			int jIm=-halfKheight, jKer=0;
			for (; jIm<halfKheight; ++jIm, ++jKer)
			{
				curCoord.y = min(max(coordinate.y+jIm,0),imageDims.y-1);
				int kIm=-halfKdepth, kKer=0;
				for (; kIm<halfKdepth; ++kIm, ++kKer)
				{
					curCoord.z = min(max(coordinate.z+kIm,0),imageDims.z-1);
					vals[iKer+jKer*kernalDims.y+kKer*kernalDims.y*kernalDims.x] = 
						imageIn[imageDims.linearAddressAt(curCoord)];
				}
			}
		}

		imageOut[imageDims.linearAddressAt(coordinate)] = cudaFindMedian(vals,kernalDims.x*kernalDims.y*kernalDims.z);
		delete vals;
	}
}

template<typename ImagePixelType1, typename ImagePixelType2>
__global__ void cudaMultAddFilter(ImagePixelType1* imageIn, ImagePixelType2* imageOut, Vec<unsigned int> hostImageDims,
	Vec<int> hostKernalDims)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<int> kernalDims = hostKernalDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		double val = 0;
		double kernFactor = 0;
		int halfKwidth = kernalDims.x/2;
		int halfKheight = kernalDims.y/2;
		int halfKdepth = kernalDims.z/2;
		int iIm=-halfKwidth, iKer=0;
		DeviceVec<unsigned int> curCoord; 
		for (; iIm<halfKwidth; ++iIm, ++iKer)
		{
			curCoord.x = min(max(coordinate.x+iIm,0),imageDims.x-1);
			int jIm=-halfKheight, jKer=0;
			for (; jIm<halfKheight; ++jIm, ++jKer)
			{
				curCoord.y = min(max(coordinate.y+jIm,0),imageDims.y-1);
				int kIm=-halfKdepth, kKer=0;
				for (; kIm<halfKdepth; ++kIm, ++kKer)
				{
					curCoord.z = min(max(coordinate.z+kIm,0),imageDims.z-1);
					kernFactor += kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x];
					val += imageIn[imageDims.linearAddressAt(coordinate)] * 
						kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x];
				}
			}
		}

		imageOut[imageDims.linearAddressAt(coordinate)] = val/kernFactor;
	}
}

template<typename ImagePixelType>
__global__ void cudaMinFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims,
	Vec<int> hostKernalDims)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<int> kernalDims = hostKernalDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		ImagePixelType minVal = imageOut[imageDims.linearAddressAt(coordinate)];
		int halfKwidth = kernalDims.x/2;
		int halfKheight = kernalDims.y/2;
		int halfKdepth = kernalDims.z/2;
		int iIm=-halfKwidth, iKer=0;
		DeviceVec<unsigned int> curCoord; 
		for (; iIm<halfKwidth; ++iIm, ++iKer)
		{
			curCoord.x = min(max(coordinate.x+iIm,0),imageDims.x-1);
			int jIm=-halfKheight, jKer=0;
			for (; jIm<halfKheight; ++jIm, ++jKer)
			{
				curCoord.y = min(max(coordinate.y+jIm,0),imageDims.y-1);
				int kIm=-halfKdepth, kKer=0;
				for (; kIm<halfKdepth; ++kIm, ++kKer)
				{
					curCoord.z = min(max(coordinate.z+kIm,0),imageDims.z-1);
					if(kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x]>0)
					{
						minVal = min((float)minVal, imageIn[imageDims.linearAddressAt(curCoord)]*
							kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x]);
					}
				}
			}
		}

		imageOut[imageDims.linearAddressAt(coordinate)] = minVal;
	}
}

template<typename ImagePixelType>
__global__ void cudaMaxFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims,
	Vec<int> hostKernalDims)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<int> kernalDims = hostKernalDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		ImagePixelType maxVal = imageOut[imageDims.linearAddressAt(coordinate)];
		int halfKwidth = kernalDims.x/2;
		int halfKheight = kernalDims.y/2;
		int halfKdepth = kernalDims.z/2;
		int iIm=-halfKwidth, iKer=0;
		DeviceVec<unsigned int> curCoord; 
		for (; iIm<halfKwidth; ++iIm, ++iKer)
		{
			curCoord.x = min(max(coordinate.x+iIm,0),imageDims.x-1);
			int jIm=-halfKheight, jKer=0;
			for (; jIm<halfKheight; ++jIm, ++jKer)
			{
				curCoord.y = min(max(coordinate.y+jIm,0),imageDims.y-1);
				int kIm=-halfKdepth, kKer=0;
				for (; kIm<halfKdepth; ++kIm, ++kKer)
				{
					curCoord.z = min(max(coordinate.z+kIm,0),imageDims.z-1);
					if(kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x]>0)
					{
						maxVal = MAX((float)maxVal, imageIn[imageDims.linearAddressAt(curCoord)]*
							kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x]);
					}
				}
			}
		}

		imageOut[imageDims.linearAddressAt(coordinate)] = maxVal;
	}
}

template<typename ImagePixelType>
__global__ void cudaHistogramCreate(ImagePixelType* imageIn, unsigned int* histogram, Vec<unsigned int> hostImageDims)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	//This code is modified from that of Sanders - Cuda by Example
	__shared__ unsigned int tempHisto[NUM_BINS];
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
__global__ void cudaHistogramCreateROI(ImagePixelType* imageIn, unsigned int* histogram, Vec<unsigned int> starts,
	Vec<unsigned int> sizes)
{
// 	//This code is modified from that of Sanders - Cuda by Example
// 	__shared__ unsigned int tempHisto[NUM_BINS];
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

__global__ void cudaNormalizeHistogram(unsigned int* histogram, double* normHistogram, Vec<unsigned int> imageDims)
{
	int x = blockIdx.x;
	normHistogram[x] = (double)(histogram[x]) / (imageDims.x*imageDims.y*imageDims.z);
}

template<typename ImagePixelType, typename ThresholdType>
__global__ void cudaThresholdImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims,
	ThresholdType threshold)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		if (imageIn[imageDims.linearAddressAt(coordinate)]>=threshold)
			imageOut[imageDims.linearAddressAt(coordinate)] = 1;
		else
			imageOut[imageDims.linearAddressAt(coordinate)] = 0;
	}
}

template<typename ImagePixelType>
__global__ void cudaFindMinMax(ImagePixelType* arrayIn, ImagePixelType* minArrayOut, ImagePixelType* maxArrayOut, unsigned int n)

{
	extern __shared__ float maxData[];
	extern __shared__ float minData[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + tid;
	unsigned int gridSize = blockDim.x*2*gridDim.x;

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

template<typename ImagePixelType, typename ThresholdType>
__global__ void cudaPolyTransferFuncImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims,
	ThresholdType a, ThresholdType b, ThresholdType c, ImagePixelType maxPixelValue, ImagePixelType minPixelValue)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<unsigned int> coordinate;
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

		ImagePixelType newPixelVal = min(maxPixelValue,max(minPixelValue, multiplier*maxPixelValue));

		imageOut[imageDims.linearAddressAt(coordinate)] = newPixelVal;
	}
}

template<typename T>
__global__ void cudaSumArray(T* arrayIn, double* arrayOut, unsigned int n)

{
	//This algorithm was used from a this website:
	// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
	// accessed 4/28/2013

	extern __shared__ T sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + tid;
	unsigned int gridSize = blockDim.x*2*gridDim.x;
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
__global__ void cudaRuduceImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageInDims,
	Vec<unsigned int> hostImageOutDims, Vec<double> hostReductions)
{
	DeviceVec<unsigned int> imageInDims = hostImageInDims;
	DeviceVec<unsigned int> imageOutDims = hostImageOutDims;
	DeviceVec<double> reductions = hostReductions;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageOutDims)
	{
		double val = 0;
		DeviceVec<unsigned int> mins, maxs;
		mins.x = coordinate.x*reductions.x;
		maxs.x = min(mins.x+reductions.x,(double)imageInDims.x);
		mins.y = coordinate.y*reductions.y;
		maxs.y = min(mins.y+reductions.y,(double)imageInDims.y);
		mins.z = coordinate.z*reductions.z;
		maxs.z = min(mins.z+reductions.z,(double)imageInDims.z);

		DeviceVec<unsigned int> currCorrd;
		for (currCorrd.x=mins.x; currCorrd.x<maxs.x; ++currCorrd.x)
		{
			for (currCorrd.y=mins.y; currCorrd.y<maxs.y; ++currCorrd.y)
			{
				for (currCorrd.y=mins.z; currCorrd.z<maxs.z; ++currCorrd.z)
				{
					val += (float)imageIn[imageInDims.linearAddressAt(currCorrd)];
				}
			}
		}

		imageOut[imageOutDims.linearAddressAt(coordinate)] = val/(maxs-mins).product();
	}
}

template<typename ImagePixelType>
__global__ void cudaMaximumIntensityProjection(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims && coordinate.z==0)
	{
		ImagePixelType maxVal = 0;
		for (; coordinate.z<imageDims.z; ++coordinate.z)
		{
			if (maxVal<imageIn[imageDims.linearAddressAt(coordinate)])
			{
				maxVal = imageIn[imageDims.linearAddressAt(coordinate)];
			}
		}

		coordinate.z = 0;
		imageOut[imageDims.linearAddressAt(coordinate)] = maxVal;
	}
}

template<typename ImagePixelType>
__global__ void cudaGetROI(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims,
	Vec<unsigned int> hostStartPos,	Vec<unsigned int> hostNewSize)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<unsigned int> newSize = hostNewSize;
	DeviceVec<unsigned int> startPos = hostStartPos;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate>=startPos && coordinate<startPos+newSize && coordinate<imageDims)
	{
		unsigned int outIndex = newSize.linearAddressAt(coordinate-startPos);
		imageOut[outIndex] = imageIn[imageDims.linearAddressAt(coordinate)];
	}
}

template<typename ImagePixelType, typename PowerType>
__global__ void cudaPow(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<unsigned int> hostImageDims, PowerType p)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
		imageOut[imageDims.linearAddressAt(coordinate)] = pow(imageIn[imageDims.linearAddressAt(coordinate)],p);
}

template<typename ImagePixelType>
__global__ void cudaUnmixing(const ImagePixelType* imageIn1, const ImagePixelType* imageIn2, ImagePixelType* imageOut1, ImagePixelType* imageOut2,
	Vec<unsigned int> hostImageDims)
{
	DeviceVec<unsigned int> imageDims = hostImageDims;
	DeviceVec<unsigned int> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageDims)
	{
		if (max(imageIn1[imageDims.linearAddressAt(coordinate)],imageIn2[imageDims.linearAddressAt(coordinate)])==
			imageIn1[imageDims.linearAddressAt(coordinate)])
		{
			imageOut1[imageDims.linearAddressAt(coordinate)] = imageIn1[imageDims.linearAddressAt(coordinate)];

			imageOut2[imageDims.linearAddressAt(coordinate)] = 
				max(0, imageIn2[imageDims.linearAddressAt(coordinate)] - imageIn1[imageDims.linearAddressAt(coordinate)]);
		}
		else if (max(imageIn1[imageDims.linearAddressAt(coordinate)],imageIn2[imageDims.linearAddressAt(coordinate)])==
			imageIn2[imageDims.linearAddressAt(coordinate)])
		{
			imageOut1[imageDims.linearAddressAt(coordinate)] = 
				max(0, imageIn1[imageDims.linearAddressAt(coordinate)] - imageIn2[imageDims.linearAddressAt(coordinate)]);

			imageOut2[imageDims.linearAddressAt(coordinate)] = imageIn2[imageDims.linearAddressAt(coordinate)];
		}
		else
		{
			imageOut1[imageDims.linearAddressAt(coordinate)] = imageIn1[imageDims.linearAddressAt(coordinate)];
			imageOut2[imageDims.linearAddressAt(coordinate)] = imageIn2[imageDims.linearAddressAt(coordinate)];
		}
	}
}
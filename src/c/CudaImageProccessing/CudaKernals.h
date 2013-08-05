#pragma once

#include "Defines.h"
#include "Vec.h"
#include "CHelpers.h"
#include "cuda_runtime.h"

__constant__ float kernal[MAX_KERNAL_DIM*MAX_KERNAL_DIM*MAX_KERNAL_DIM];

template<typename ImagePixelType>
__global__ void cudaMeanFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, Vec<int> kernalDims)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z<imageDims.z)
	{
		double val = 0;
		int halfKwidth = kernalDims.x/2;
		int halfKheight = kernalDims.y/2;
		int halfKdepth = kernalDims.z/2;
		int iIm=-halfKwidth, iKer=0;
		for (; iIm<halfKwidth; ++iIm, ++iKer)
		{
			int imXcoor = min(max(x+iIm,0),imageDims.x-1);
			int jIm=-halfKheight, jKer=0;
			for (; jIm<halfKheight; ++jIm, ++jKer)
			{
				int imYcoor = min(max(y+jIm,0),imageDims.y-1);
				int kIm=-halfKdepth, kKer=0;
				for (; kIm<halfKdepth; ++kIm, ++kKer)
				{
					int imZcoor = min(max(z+kIm,0),imageDims.z-1);
					val += imageIn[imXcoor + imYcoor*imageDims.x + imZcoor*imageDims.y*imageDims.x];
				}
			}
		}

		imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = val/(kernalDims.x*kernalDims.y*kernalDims.z);
	}
}

template<typename ImagePixelType, typename FactorType>
__global__ void cudaMultiplyImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, FactorType factor,
	ImagePixelType minValue, ImagePixelType maxValue)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z<imageDims.z)
	{
		imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = min(maxValue,max(minValue,
			factor * imageIn[x+y*imageDims.x+z*imageDims.y*imageDims.x]));
	}
}

template<typename ImagePixelType1, typename ImagePixelType2, typename ImagePixelType3, typename FactorType>
__global__ void cudaAddTwoImagesWithFactor(ImagePixelType1* imageIn1, ImagePixelType2* imageIn2, ImagePixelType3* imageOut,
	Vec<int> imageDims, FactorType factor)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z<imageDims.z)
	{
		imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = 
			imageIn1[x+y*imageDims.x+z*imageDims.y*imageDims.x] - factor*imageIn2[x+y*imageDims.x+z*imageDims.y*imageDims.x];
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
__global__ void cudaMedianFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, Vec<int> kernalDims)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z<imageDims.z)
	{
		ImagePixelType* vals = new ImagePixelType[kernalDims.x*kernalDims.y*kernalDims.z];
		int halfKwidth = kernalDims.x/2;
		int halfKheight = kernalDims.y/2;
		int halfKdepth = kernalDims.z/2;
		int iIm=-halfKwidth, iKer=0;
		for (; iIm<halfKwidth; ++iIm, ++iKer)
		{
			int imXcoor = min(max(x+iIm,0),imageDims.x-1);
			int jIm=-halfKheight, jKer=0;
			for (; jIm<halfKheight; ++jIm, ++jKer)
			{
				int imYcoor = min(max(y+jIm,0),imageDims.y-1);
				int kIm=-halfKdepth, kKer=0;
				for (; kIm<halfKdepth; ++kIm, ++kKer)
				{
					int imZcoor = min(max(z+kIm,0),imageDims.z-1);
					vals[iKer+jKer*kernalDims.y+kKer*kernalDims.y*kernalDims.x] = 
						imageIn[imXcoor + imYcoor*imageDims.x + imZcoor*imageDims.y*imageDims.x];
				}
			}
		}

		imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = findMedian(vals,kernalDims.x*kernalDims.y*kernalDims.z);
		delete vals;
	}
}

template<typename ImagePixelType1, typename ImagePixelType2>
__global__ void cudaMultAddFilter(ImagePixelType1* imageIn, ImagePixelType2* imageOut, Vec<int> imageDims, Vec<int> kernalDims)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z<imageDims.z)
	{
		double val = 0;
		int halfKwidth = kernalDims.x/2;
		int halfKheight = kernalDims.y/2;
		int halfKdepth = kernalDims.z/2;
		int iIm=-halfKwidth, iKer=0;
		double kernFactor = 0;
		for (; iIm<halfKwidth; ++iIm, ++iKer)
		{
			int imXcoor = min(max(x+iIm,0),imageDims.x-1);
			int jIm=-halfKheight, jKer=0;
			for (; jIm<halfKheight; ++jIm, ++jKer)
			{
				int imYcoor = min(max(y+jIm,0),imageDims.y-1);
				int kIm=-halfKdepth, kKer=0;
				for (; kIm<halfKdepth; ++kIm, ++kKer)
				{
					int imZcoor = min(max(z+kIm,0),imageDims.z-1);
					kernFactor += kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x];
					val += imageIn[imXcoor + imYcoor*imageDims.x + imZcoor*imageDims.y*imageDims.x] * 
						kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x];
				}
			}
		}

		imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = val/kernFactor;
	}
}

template<typename ImagePixelType>
__global__ void cudaMinFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, Vec<int> kernalDims)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z<imageDims.z)
	{
		ImagePixelType minVal = imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.z];
		int halfKwidth = kernalDims.x/2;
		int halfKheight = kernalDims.y/2;
		int halfKdepth = kernalDims.z/2;
		int iIm=-halfKwidth, iKer=0;
		for (; iIm<halfKwidth; ++iIm, ++iKer)
		{
			int imXcoor = min(max(x+iIm,0),imageDims.x-1);
			int jIm=-halfKheight, jKer=0;
			for (; jIm<halfKheight; ++jIm, ++jKer)
			{
				int imYcoor = min(max(y+jIm,0),imageDims.y-1);
				int kIm=-halfKdepth, kKer=0;
				for (; kIm<halfKdepth; ++kIm, ++kKer)
				{
					int imZcoor = min(max(z+kIm,0),imageDims.z-1);
					if(kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x]>0)
					{
						minVal = min((float)minVal, imageIn[imXcoor + imYcoor*imageDims.x + imZcoor*imageDims.y*imageDims.x]*
							kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x]);
					}
				}
			}
		}

		imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = minVal;
	}
}

template<typename ImagePixelType>
__global__ void cudaMaxFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, Vec<int> kernalDims)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z<imageDims.z)
	{
		ImagePixelType maxVal = imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x];
		int halfKwidth = kernalDims.x/2;
		int halfKheight = kernalDims.y/2;
		int halfKdepth = kernalDims.z/2;
		int iIm=-halfKwidth, iKer=0;
		for (; iIm<halfKwidth; ++iIm, ++iKer)
		{
			int imXcoor = min(max(x+iIm,0),imageDims.x-1);
			int jIm=-halfKheight, jKer=0;
			for (; jIm<halfKheight; ++jIm, ++jKer)
			{
				int imYcoor = min(max(y+jIm,0),imageDims.y-1);
				int kIm=-halfKdepth, kKer=0;
				for (; kIm<halfKdepth; ++kIm, ++kKer)
				{
					int imZcoor = min(max(z+kIm,0),imageDims.z-1);
					if(kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x]>0)
					{
						maxVal = max((float)maxVal, imageIn[imXcoor + imYcoor*imageDims.x + imZcoor*imageDims.y*imageDims.x]*
							kernal[iKer+jKer*kernalDims.x+kKer*kernalDims.y*kernalDims.x]);
					}
				}
			}
		}

		imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = maxVal;
	}
}

template<typename ImagePixelType>
__global__ void cudaHistogramCreate(ImagePixelType* imageIn, unsigned int* histogram, Vec<int> imageDims)
{
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

__global__ void cudaNormalizeHistogram(unsigned int* histogram, double* normHistogram, Vec<int> imageDims)
{
	int x = blockIdx.x;
	normHistogram[x] = (double)(histogram[x]) / (imageDims.x*imageDims.y*imageDims.z);
}

template<typename ImagePixelType, typename ThresholdType>
__global__ void cudaThresholdImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, ThresholdType threshold)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z<imageDims.z)
	{
		if (imageIn[x+y*imageDims.x+z*imageDims.y*imageDims.x]>=threshold)
			imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = 1;
		else
			imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = 0;
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
__global__ void cudaPolyTransferFuncImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, ThresholdType a,
	ThresholdType b, ThresholdType c, ThresholdType maxPixelValue, ThresholdType minPixelValue)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z<imageDims.z)
	{
		double pixVal = (double)imageIn[x+y*imageDims.x+z*imageDims.y*imageDims.x] / maxPixelValue;
		double multiplier = a*pixVal*pixVal + b*pixVal + c;
		if (multiplier<0)
			multiplier = 0;
		if (multiplier>1)
			multiplier = 1;

		ImagePixelType newPixelVal = min(maxPixelValue,max(minPixelValue,multiplier * maxPixelValue));

		imageOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = newPixelVal;
	}
}

template<typename ImagePixelType, typename ThresholdType>
__global__ void cudaGetCoordinates(ImagePixelType* imageIn, Vec<int>* coordinatesOut, Vec<int> imageDims, ThresholdType threshold)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z<imageDims.z)
	{
		Vec<int> coord;
		if (imageIn[x+y*imageDims.x+z*imageDims.y*imageDims.x]>threshold)
		{
			coord = Vec<int>(x,y,z);
		}
		else
		{
			coord = Vec<int>(-1,-1,-1);
		}

		coordinatesOut[x+y*imageDims.x+z*imageDims.y*imageDims.x] = coord;
	}
}

__global__ void cudaFillCoordinates(Vec<int>* coordinatesIn, Vec<int>* coordinatesOut, Vec<int> imageDims, int overDimension);

template<typename T>
__global__ void cudaReduceArray(T* arrayIn, T* arrayOut, unsigned int n)

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
__global__ void cudaRuduceImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageInDims, Vec<int> imageOutDims, Vec<double> reductions)
{
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageOutDims.x && y<imageOutDims.y && z<imageOutDims.z)
	{
		double val = 0;
		unsigned int xMin = x*reductions.x;
		unsigned int xMax = min(x*reductions.x+reductions.x,(double)imageInDims.x);
		unsigned int yMin = y*reductions.y;
		unsigned int yMax = min(y*reductions.y+reductions.y,(double)imageInDims.y);
		unsigned int zMin = z*reductions.z;
		unsigned int zMax = min(z*reductions.z+reductions.z,(double)imageInDims.z);

		for (unsigned int i=xMin; i<xMax; ++i)
		{
			for (unsigned int j=yMin; j<yMax; ++j)
			{
				for (unsigned int k=zMin; k<zMax; ++k)
					//center imageIn[x+y*imageWidth]
					val += (float)imageIn[i+j*imageInDims.x+k*imageInDims.y*imageInDims.x];
			}
		}

		imageOut[x+y*imageOutDims.x+z*imageOutDims.y*imageOutDims.x] = val/((xMax-xMin)*(yMax-yMin)*(zMax-zMin));
	}
}

template<typename ImagePixelType>
__global__ void cudaMaximumIntensityProjection(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageDims.x && y<imageDims.y && z==0)
	{
		ImagePixelType maxVal = 0;
		for (int depth=0; depth<imageDims.z; ++depth)
		{
			if (maxVal<imageIn[x+y*imageDims.x+depth*imageDims.y*imageDims.x])
			{
				maxVal = imageIn[x+y*imageDims.x+depth*imageDims.y*imageDims.x];
			}
		}

		imageOut[x+y*imageDims.x] = maxVal;
	}
}

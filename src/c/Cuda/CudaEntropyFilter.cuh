#pragma once

#include "CudaImageContainer.cuh"
#include "Vec.h"
#include <vector>
#include "CHelpers.h"
#include "CudaUtilities.cuh"
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"
#include "CudaGetMinMax.cuh"

#include <functional>

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

class HistoShared
{
public:
	inline __host__ __device__ HistoShared() {}
	inline __host__ __device__ int operator()(int n)
	{
		n = n*256*sizeof(short);
		return n;
	}
};

template <class PixelType>
__global__ void cudaEntropyFilter(CudaImageContainer<PixelType> imageIn, CudaImageContainer<double> imageOut, Vec<size_t> hostKernelDims, PixelType minVal, PixelType maxVal)
{
	//extern __shared__ unsigned short histo[];

	Vec<int> kernelDims = hostKernelDims;
	Vec<int> coordinateOut;
	coordinateOut.x = threadIdx.x+blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y+blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z+blockIdx.z * blockDim.z;

	//unsigned short* lclHisto = (threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.y*blockDim.x)*256 +histo;
	unsigned short lclHisto[256];
	double binWidth = (double)(maxVal-minVal+1)/256.0f;
	for (int i=0; i<255; ++i)
	{
		lclHisto[i] = 0;
	}

	if(coordinateOut<imageOut.getDims())
	{
		double val = 0;
		Vec<float> inputCenter = Vec<float>(coordinateOut+0.5f);
		Vec<float> kernelCenter = Vec<float>(kernelDims-1)/2.0f;
		Vec<int> kernelStart(0, 0, 0);
		Vec<int> kernelEnd(0, 0, 0);

		Vec<float> neighborhoodStart = inputCenter-Vec<float>(kernelDims)/2.0f;
		// if the input start position is negative, we need to start further in on the kernel
		kernelStart.x = (neighborhoodStart.x>=0.0f) ? (0) : (ceil(-neighborhoodStart.x));
		kernelStart.y = (neighborhoodStart.y>=0.0f) ? (0) : (ceil(-neighborhoodStart.y));
		kernelStart.z = (neighborhoodStart.z>=0.0f) ? (0) : (ceil(-neighborhoodStart.z));
		neighborhoodStart = Vec<float>::max(Vec<float>(0.0f, 0.0f, 0.0f), neighborhoodStart);

		// This is the last place to visit in the input (inclusive)
		Vec<float> neighborhoodEnd = inputCenter+(Vec<float>(kernelDims)/2.0f);
		// if the input end position is outside the image, we need to end earlier in on the kernel
		kernelEnd.x = (neighborhoodEnd.x<=imageIn.getDims().x) ? (kernelDims.x) :
			(kernelDims.x-(neighborhoodEnd.x-imageIn.getDims().x));// will floor to int value
		kernelEnd.y = (neighborhoodEnd.y<=imageIn.getDims().y) ? (kernelDims.y) :
			(kernelDims.y-(neighborhoodEnd.y-imageIn.getDims().y));// will floor to int value
		kernelEnd.z = (neighborhoodEnd.z<=imageIn.getDims().z) ? (kernelDims.z) :
			(kernelDims.z-(neighborhoodEnd.z-imageIn.getDims().z));// will floor to int value

		neighborhoodEnd = Vec<float>::min(imageIn.getDims(), neighborhoodEnd);

		Vec<int> curKernelPos(0, 0, 0);
		Vec<int> curInPos = neighborhoodStart;
		double numVals = 0;
		for(curKernelPos.z = kernelStart.z; curKernelPos.z<kernelEnd.z; ++curKernelPos.z)
		{
			curInPos.z = neighborhoodStart.z+curKernelPos.z;
			for(curKernelPos.y = kernelStart.y; curKernelPos.y<kernelEnd.y; ++curKernelPos.y)
			{
				curInPos.y = neighborhoodStart.y+curKernelPos.y;
				for(curKernelPos.x = kernelStart.x; curKernelPos.x<kernelEnd.x; ++curKernelPos.x)
				{
					if (cudaConstKernel[hostKernelDims.linearAddressAt(curKernelPos)]>0)
					{
						curInPos.x = neighborhoodStart.x+curKernelPos.x;
						PixelType imVal = imageIn(curInPos);
						int binNum = floor((double)(imVal-minVal)/binWidth);
						++(lclHisto[binNum]);
						++numVals;
					}
				}
			}
		}

		for(int i = 0; i<255; ++i)
		{
			double hVal = lclHisto[i]/numVals;
			if (hVal>0)
				val = val+hVal*log2(hVal);
		}


		imageOut(coordinateOut) = -val;
	}
}

#pragma optimize("",off)
template <class PixelType>
double* cEntropyFilter(const PixelType* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, double** imageOut = NULL, int device = 0)
{
	cudaSetDevice(device);
	double* imOut = setUpOutIm(dims, imageOut);

	PixelType minVal = 0;
	PixelType maxVal = 0;

	cGetMinMax(imageIn, dims, minVal, maxVal, device);

	if(kernel==NULL)
	{
		kernelDims = kernelDims.clamp(Vec<size_t>(1, 1, 1), dims);
		float* ones = new float[kernelDims.product()];
		for(int i = 0; i<kernelDims.product(); ++i)
			ones[i] = 1.0f;

		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, ones, sizeof(double)*kernelDims.product()));
		delete[] ones;
	} else
	{
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, kernel, sizeof(double)*kernelDims.product()));
	}

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem, &total);

	HistoShared hitoMemObj;
	//int blockSize = getKernelMaxThreadsSharedMem(cudaEntropyFilter<PixelType>, hitoMemObj);
	int blockSize = getKernelMaxThreads(cudaEntropyFilter<PixelType>);

	double inOutSize = (double)(sizeof(PixelType)+sizeof(double));
	double inputPrcnt = sizeof(PixelType)/inOutSize;

	std::vector<ImageChunk> inChunks = calculateBuffers<PixelType>(dims, 1, (size_t)(availMem*MAX_MEM_AVAIL*inputPrcnt), props, kernelDims, blockSize);
	std::vector<ImageChunk> outChunks(inChunks);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(inChunks, maxDeviceDims);
	CudaDeviceImages<PixelType> deviceInImages(1, maxDeviceDims, device);
	CudaDeviceImages<double> deviceOutImages(1, maxDeviceDims, device);

	std::vector<ImageChunk>::iterator inIt = inChunks.begin();
	std::vector<ImageChunk>::iterator outIt = outChunks.begin();

	while(inIt!=inChunks.end() && outIt!=outChunks.end())
	{
		inIt->sendROI(imageIn, dims, deviceInImages.getCurBuffer());
		deviceInImages.setNextDims(inIt->getFullChunkSize());

		//int sharedMemSize = hitoMemObj((inIt->threads.x * inIt->threads.y * inIt->threads.z));

		//cudaEntropyFilter<<<inIt->blocks, inIt->threads, sharedMemSize>>>(*(deviceInImages.getCurBuffer()), *(deviceOutImages.getCurBuffer()), kernelDims, minVal, maxVal);
		cudaEntropyFilter<<<inIt->blocks, inIt->threads>>>(*(deviceInImages.getCurBuffer()), *(deviceOutImages.getCurBuffer()), kernelDims, minVal, maxVal);
		DEBUG_KERNEL_CHECK();

		outIt->retriveROI(imOut, dims, deviceOutImages.getCurBuffer());

		++inIt;
		++outIt;
	}

	return imOut;
}
#pragma optimize("",on)
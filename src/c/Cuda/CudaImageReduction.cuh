#pragma once

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
__global__ void cudaMeanImageReduction(CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
									   Vec<size_t> hostReductions)
{
	Vec<size_t> reductions = hostReductions;
	Vec<size_t> coordinateOut;
	coordinateOut.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinateOut<imageOut.getDims())
	{
		double kernelVolume = 0;
		double val = 0;
		Vec<size_t> mins(coordinateOut*reductions);
		Vec<size_t> maxs = Vec<size_t>::min(mins+reductions, imageIn.getDims());

		Vec<size_t> currCorrdIn(mins);
		for (currCorrdIn.z=mins.z; currCorrdIn.z<maxs.z; ++currCorrdIn.z)
		{
			for (currCorrdIn.y=mins.y; currCorrdIn.y<maxs.y; ++currCorrdIn.y)
			{
				for (currCorrdIn.x=mins.x; currCorrdIn.x<maxs.x; ++currCorrdIn.x)
				{
					val += (double)imageIn[currCorrdIn];
					++kernelVolume;
				}
			}
		}

		imageOut[coordinateOut] = val/kernelVolume;
	}
}

template <class PixelType>
__global__ void cudaMedianImageReduction( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
										 Vec<size_t> hostReductions)
{
	extern __shared__ unsigned char sharedMem[];
	PixelType* vals = (PixelType*)sharedMem;
	Vec<size_t> reductions = hostReductions;
	Vec<size_t> coordinateOut;
	coordinateOut.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
	offset *=  reductions.product();

	if (coordinateOut<imageOut.getDims())
	{
		int kernelVolume = 0;
		Vec<size_t> mins(coordinateOut*Vec<size_t>(reductions));
		Vec<size_t> maxs = Vec<size_t>::min(mins+reductions, imageIn.getDims());

		Vec<size_t> currCorrdIn(mins);
		for (currCorrdIn.z=mins.z; currCorrdIn.z<maxs.z; ++currCorrdIn.z)
		{
			for (currCorrdIn.y=mins.y; currCorrdIn.y<maxs.y; ++currCorrdIn.y)
			{
				for (currCorrdIn.x=mins.x; currCorrdIn.x<maxs.x; ++currCorrdIn.x)
				{
					vals[offset+kernelVolume] = (double)imageIn[currCorrdIn];
					++kernelVolume;
				}
			}
		}
		imageOut[coordinateOut] = (PixelType)cudaFindMedian(vals+offset,kernelVolume);
	}
	__syncthreads();
}

template <class PixelType>
__global__ void cudaMaxImageReduction(CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
									   Vec<size_t> hostReductions, PixelType minVal)
{
	Vec<size_t> reductions = hostReductions;
	Vec<size_t> coordinateOut;
	coordinateOut.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinateOut<imageOut.getDims())
	{
		PixelType val = minVal;
		Vec<size_t> mins(coordinateOut*reductions);
		Vec<size_t> maxs = Vec<size_t>::min(mins+reductions, imageIn.getDims());

		Vec<size_t> currCorrdIn(mins);
		for (currCorrdIn.z=mins.z; currCorrdIn.z<maxs.z; ++currCorrdIn.z)
		{
			for (currCorrdIn.y=mins.y; currCorrdIn.y<maxs.y; ++currCorrdIn.y)
			{
				for (currCorrdIn.x=mins.x; currCorrdIn.x<maxs.x; ++currCorrdIn.x)
				{
					if (val<imageIn[currCorrdIn])
						val = imageIn[currCorrdIn];
				}
			}
		}

		imageOut[coordinateOut] = val;
	}
}

template <class PixelType>
__global__ void cudaMinImageReduction(CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
									  Vec<size_t> hostReductions, PixelType maxVal)
{
	Vec<size_t> reductions = hostReductions;
	Vec<size_t> coordinateOut;
	coordinateOut.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinateOut<imageOut.getDims())
	{
		PixelType val = maxVal;
		Vec<size_t> mins(coordinateOut*reductions);
		Vec<size_t> maxs = Vec<size_t>::min(mins+reductions, imageIn.getDims());

		Vec<size_t> currCorrdIn(mins);
		for (currCorrdIn.z=mins.z; currCorrdIn.z<maxs.z; ++currCorrdIn.z)
		{
			for (currCorrdIn.y=mins.y; currCorrdIn.y<maxs.y; ++currCorrdIn.y)
			{
				for (currCorrdIn.x=mins.x; currCorrdIn.x<maxs.x; ++currCorrdIn.x)
				{
					if (val>imageIn[currCorrdIn])
						val = imageIn[currCorrdIn];
				}
			}
		}

		imageOut[coordinateOut] = val;
	}
}

template <class PixelType>
__global__ void cudaGausImageReduction(CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
									   Vec<int> hostReductions, Vec<int> hostKernalDims, PixelType minVal, PixelType maxVal)
{
	Vec<int> reductions = hostReductions;
	Vec<int> coordinateOut;
	coordinateOut.x = threadIdx.x+blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y+blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z+blockIdx.z * blockDim.z;

	if(coordinateOut<imageOut.getDims())
	{
		double val = 0;
		double kernFactor = 0;

		Vec<int> coordinateIn = coordinateOut*reductions+(reductions+1)/2.0;

		Vec<int> kernelDims = hostKernalDims;

		Vec<int> startLimit = coordinateIn-kernelDims/2;
		Vec<int> endLimit = coordinateIn+(kernelDims+1)/2;
		Vec<int> kernelStart(Vec<int>::max(-startLimit, Vec<int>(0, 0, 0)));

		startLimit = Vec<int>::max(startLimit, Vec<int>(0, 0, 0));
		endLimit = Vec<int>::min(endLimit, imageIn.getDims());

		Vec<int> imageStart(coordinateIn-(kernelDims/2)+kernelStart);
		Vec<int> iterationEnd(endLimit-startLimit+1);

		Vec<int> centerOffset(0, 0, 0);
		for(centerOffset.z = 0; centerOffset.z<iterationEnd.z; ++centerOffset.z)
		{
			for(centerOffset.y = 0; centerOffset.y<iterationEnd.y; ++centerOffset.y)
			{
				for(centerOffset.x = 0; centerOffset.x<iterationEnd.x; ++centerOffset.x)
				{
					double kernVal = double(cudaConstKernel[kernelDims.linearAddressAt(kernelStart+centerOffset)]);

					kernFactor += kernVal;
					val += double((imageIn[imageStart+centerOffset]) * kernVal);
				}
			}
		}

		val = val/kernFactor;
		val = (val<minVal) ? (minVal) : (val);
		val = (val>maxVal) ? (maxVal) : (val);

		imageOut[coordinateOut] = (PixelType)val;
	}
}


template <class PixelType>
PixelType* cReduceImage(const PixelType* imageIn, Vec<size_t> dims, Vec<double> reductions, Vec<size_t>& reducedDims, 
					   ReductionMethods method=REDUC_MEAN, PixelType** imageOut=NULL, int device=0)
{
	cudaSetDevice(device);
	reductions = reductions.clamp(Vec<size_t>(1,1,1),dims);
	reducedDims = Vec<size_t>(Vec<double>(dims) / reductions);
	PixelType* reducedImage;
	if (imageOut==NULL)
		reducedImage = new PixelType[reducedDims.product()];
	else
		reducedImage = *imageOut;

	double ratio = (double)reducedDims.product() / dims.product();

	if (ratio==1.0)
	{
		memcpy(reducedImage,imageIn,sizeof(PixelType)*reducedDims.product());
		return reducedImage;
	}

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t memAvail, total;
	cudaMemGetInfo(&memAvail,&total);

	std::vector<ImageChunk> orgChunks;
	int numThreads = props.maxThreadsPerBlock;
	Vec<int> gaussIterations(0, 0, 0);
	float* hostKernel;
	Vec<size_t> sizeconstKernelDims = Vec<size_t>(0, 0, 0);

	if (method==REDUC_MEDIAN)
	{
		size_t sizeOfsharedMem = (size_t)(reductions.product())*sizeof(PixelType);
		numThreads = (int)floor((double)props.sharedMemPerBlock/(double)sizeOfsharedMem);
		numThreads = (props.maxThreadsPerBlock>numThreads) ? numThreads : props.maxThreadsPerBlock;
		if(numThreads<1)
			throw std::runtime_error("Median neighborhood is too large to fit in shared memory on the GPU!");

		orgChunks = calculateBuffers<PixelType>(dims, 1, (size_t)(memAvail*MAX_MEM_AVAIL*(1-ratio)), props, reductions, numThreads);
	}
	else if(method==REDUC_GAUS)
	{
		Vec<float> sigmas = Vec<float>((reductions-1)*0.5f);

		sizeconstKernelDims = createGaussianKernelFull(sigmas, &hostKernel);
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*sizeconstKernelDims.product()));

		orgChunks = calculateBuffers<PixelType>(dims, 1, (size_t)(memAvail*MAX_MEM_AVAIL*(1-ratio)), props, sizeconstKernelDims);
	}
	else
	{
		orgChunks = calculateBuffers<PixelType>(dims, 1, (size_t)(memAvail*MAX_MEM_AVAIL*(1-ratio)), props, reductions);
	}

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(orgChunks, maxDeviceDims);
	CudaImageContainerClean<PixelType>* deviceImageIn = new CudaImageContainerClean<PixelType>(maxDeviceDims,device);

	cudaMemGetInfo(&memAvail, &total);
	std::vector<ImageChunk> reducedChunks = calculateBuffers<PixelType>(reducedDims, 1, (size_t)(memAvail*MAX_MEM_AVAIL), props, sizeconstKernelDims, numThreads);
	setMaxDeviceDims(reducedChunks, maxDeviceDims);
	CudaImageContainerClean<PixelType>* deviceImageOut = new CudaImageContainerClean<PixelType>(maxDeviceDims,device);

	std::vector<ImageChunk>::iterator orgIt = orgChunks.begin();
	std::vector<ImageChunk>::iterator reducedIt = reducedChunks.begin();

	size_t sharedMemorysize = 0;

	while (orgIt!=orgChunks.end() && reducedIt!=reducedChunks.end())
	{
		orgIt->sendROI(imageIn,dims,deviceImageIn);
		deviceImageOut->setDims(reducedIt->getFullChunkSize());

		switch (method)
		{
		case REDUC_MEAN:
			cudaMeanImageReduction<<<reducedIt->blocks,reducedIt->threads>>>(*deviceImageIn, *deviceImageOut, reductions);
			break;
		case REDUC_MEDIAN:
			sharedMemorysize = (size_t)(reductions.product())*sizeof(PixelType) * reducedIt->threads.x * reducedIt->threads.y * reducedIt->threads.z;
			cudaMedianImageReduction<<<reducedIt->blocks,reducedIt->threads,sharedMemorysize>>>(*deviceImageIn, *deviceImageOut, reductions);
			break;
		case REDUC_MIN:
			cudaMinImageReduction<<<reducedIt->blocks,reducedIt->threads>>>(*deviceImageIn, *deviceImageOut, reductions,
				std::numeric_limits<PixelType>::max());
			break;
		case REDUC_MAX:
			cudaMaxImageReduction<<<reducedIt->blocks,reducedIt->threads>>>(*deviceImageIn, *deviceImageOut, reductions,
				std::numeric_limits<PixelType>::lowest());
			break;
		case REDUC_GAUS:
			cudaGausImageReduction<<<reducedIt->blocks, reducedIt->threads>>>(*deviceImageIn, *deviceImageOut, reductions, sizeconstKernelDims,
																			  std::numeric_limits<PixelType>::lowest(), std::numeric_limits<PixelType>::max());
			break;
		default:
			break;
		}
		DEBUG_KERNEL_CHECK();

		reducedIt->retriveROI(reducedImage,reducedDims,deviceImageOut);

		++orgIt;
		++reducedIt;
	}

	delete deviceImageIn;
	delete deviceImageOut;

	return reducedImage;
}
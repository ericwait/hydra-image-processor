#pragma once

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

template <class PixelType>
__global__ void cudaMeanImageReduction(CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
									   Vec<size_t> hostReductions)
{
	DeviceVec<size_t> reductions = hostReductions;
	DeviceVec<size_t> coordinateOut;
	coordinateOut.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinateOut<imageOut.getDeviceDims())
	{
		double kernelVolume = 0;
		double val = 0;
		DeviceVec<size_t> mins(coordinateOut*reductions);
		DeviceVec<size_t> maxs = DeviceVec<size_t>::min(mins+reductions, imageIn.getDeviceDims());

		DeviceVec<size_t> currCorrdIn(mins);
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
	extern __shared__ DevicePixelType vals[];
	DeviceVec<size_t> reductions = hostReductions;
	DeviceVec<size_t> coordinateOut;
	coordinateOut.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
	offset *=  reductions.product();

	if (coordinateOut<imageOut.getDeviceDims())
	{
		int kernelVolume = 0;
		DeviceVec<size_t> mins(coordinateOut*DeviceVec<size_t>(reductions));
		DeviceVec<size_t> maxs = DeviceVec<size_t>::min(mins+reductions, imageIn.getDeviceDims());

		DeviceVec<size_t> currCorrdIn(mins);
		for (currCorrdIn.z=mins.z; currCorrdIn.z<maxs.z; ++currCorrdIn.z)
		{
			for (currCorrdIn.y=mins.y; currCorrdIn.y<maxs.y; ++currCorrdIn.y)
			{
				for (currCorrdIn.x=mins.x; currCorrdIn.x<maxs.x; ++currCorrdIn.x)
				{
					vals[offset+kernelVolume] = imageIn[currCorrdIn];
					++kernelVolume;
				}
			}
		}
		imageOut[coordinateOut] = cudaFindMedian(vals+offset,kernelVolume);
	}
	__syncthreads();
}

template <class PixelType>
__global__ void cudaMaxImageReduction(CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
									   Vec<size_t> hostReductions, PixelType minVal)
{
	DeviceVec<size_t> reductions = hostReductions;
	DeviceVec<size_t> coordinateOut;
	coordinateOut.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinateOut<imageOut.getDeviceDims())
	{
		PixelType val = minVal;
		DeviceVec<size_t> mins(coordinateOut*reductions);
		DeviceVec<size_t> maxs = DeviceVec<size_t>::min(mins+reductions, imageIn.getDeviceDims());

		DeviceVec<size_t> currCorrdIn(mins);
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
	DeviceVec<size_t> reductions = hostReductions;
	DeviceVec<size_t> coordinateOut;
	coordinateOut.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinateOut<imageOut.getDeviceDims())
	{
		PixelType val = maxVal;
		DeviceVec<size_t> mins(coordinateOut*reductions);
		DeviceVec<size_t> maxs = DeviceVec<size_t>::min(mins+reductions, imageIn.getDeviceDims());

		DeviceVec<size_t> currCorrdIn(mins);
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
PixelType* reduceImage(const PixelType* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, 
					   ReductionMethods method=REDUC_MEAN, PixelType** imageOut=NULL, int device=0)
{
	reductions = reductions.clamp(Vec<size_t>(1,1,1),dims);
	reducedDims = dims / reductions;
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

	std::vector<ImageChunk> orgChunks = calculateBuffers<PixelType>(dims,1,(size_t)(memAvail*MAX_MEM_AVAIL*(1-ratio)),props,reductions);
	std::vector<ImageChunk> reducedChunks = orgChunks;

	for (std::vector<ImageChunk>::iterator it=reducedChunks.begin(); it!=reducedChunks.end(); ++it)
	{
		it->imageStart = it->imageROIstart/reductions;
		it->chunkROIstart = Vec<size_t>(0,0,0);
		it->imageROIstart = it->imageROIstart/reductions;
		it->imageEnd = it->imageROIend/reductions;
		it->imageROIend = it->imageROIend/reductions;
		it->chunkROIend = it->imageEnd-it->imageStart;

		calcBlockThread(it->getFullChunkSize(),props,it->blocks,it->threads);
	}

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(orgChunks, maxDeviceDims);

	CudaImageContainerClean<PixelType>* deviceImageIn = new CudaImageContainerClean<PixelType>(maxDeviceDims,device);

	setMaxDeviceDims(reducedChunks, maxDeviceDims);
	CudaImageContainerClean<PixelType>* deviceImageOut = new CudaImageContainerClean<PixelType>(maxDeviceDims,device);

	std::vector<ImageChunk>::iterator orgIt = orgChunks.begin();
	std::vector<ImageChunk>::iterator reducedIt = reducedChunks.begin();

	while (orgIt!=orgChunks.end() && reducedIt!=reducedChunks.end())
	{
		orgIt->sendROI(imageIn,dims,deviceImageIn);
		deviceImageOut->setDims(reducedIt->getFullChunkSize());

		dim3 blocks(reducedIt->blocks);
		dim3 threads(reducedIt->threads);
		double threadVolume = threads.x * threads.y * threads.z;
		double newThreadVolume = (double)props.sharedMemPerBlock/(sizeof(PixelType)*reductions.product());

		if (newThreadVolume<threadVolume)
		{
			double alpha = pow(threadVolume/newThreadVolume,1.0/3.0);
			threads.x = (unsigned int)(threads.x / alpha);
			threads.y = (unsigned int)(threads.y / alpha);
			threads.z = (unsigned int)(threads.z / alpha);
			threads.x = (threads.x>0) ? (threads.x) : (1);
			threads.y = (threads.y>0) ? (threads.y) : (1);
			threads.z = (threads.z>0) ? (threads.z) : (1);

			if (threads.x*threads.y*threads.z>(unsigned int)props.maxThreadsPerBlock)
			{
				unsigned int maxThreads = (unsigned int)pow(props.maxThreadsPerBlock,1.0/3.0);
				threads.x = maxThreads;
				threads.y = maxThreads;
				threads.z = maxThreads;
			}


			blocks.x = (unsigned int)ceil((double)reducedIt->getFullChunkSize().x / threads.x);
			blocks.y = (unsigned int)ceil((double)reducedIt->getFullChunkSize().y / threads.y);
			blocks.z = (unsigned int)ceil((double)reducedIt->getFullChunkSize().z / threads.z);
		}

		size_t sharedMemorysize = reductions.product()*sizeof(PixelType) * threads.x * threads.y * threads.z;

		switch (method)
		{
		case REDUC_MEAN:
			cudaMeanImageReduction<<<blocks,threads,sharedMemorysize>>>(*deviceImageIn, *deviceImageOut, reductions);
			break;
		case REDUC_MEDIAN:
			cudaMedianImageReduction<<<blocks,threads,sharedMemorysize>>>(*deviceImageIn, *deviceImageOut, reductions);
			break;
		case REDUC_MIN:
			cudaMinImageReduction<<<blocks,threads,sharedMemorysize>>>(*deviceImageIn, *deviceImageOut, reductions,
				std::numeric_limits<PixelType>::max());
			break;
		case REDUC_MAX:
			cudaMaxImageReduction<<<blocks,threads,sharedMemorysize>>>(*deviceImageIn, *deviceImageOut, reductions,
				std::numeric_limits<PixelType>::lowest());
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
#include "KernelIterator.cuh"

#include <limits.h>
#include <float.h>

__device__ KernelIterator::KernelIterator(Vec<size_t> inputPos, ImageDimensions inputSize, Vec<size_t> kernelSize)
{
	Vec<float> inputCoordinate(inputPos);

	// If kernel size is odd, this will be integer, else we will get a real number
	Vec<float> kernelHalfSize = (Vec<float>(kernelSize)-1)/2.0f;
	
	// This puts the first kernel position in the center of the voxel
	// i.e. 1 is the second voxel position but 1.5 is an interpolated position between the first and second positions
	inputStartCoordinate = inputCoordinate - kernelHalfSize;

	// The negative here will make the kernel start w/in if it went out of bounds
	// e.g. a -1 will start at the kernel's second position 
	kernelStartIdx = Vec<size_t>::max(Vec<size_t>(0), Vec<size_t>(-inputStartCoordinate));

	// This should be the last place in the image that will be considered
	Vec<float> imageEndCoordinate = inputStartCoordinate + Vec<float>(kernelSize-1);
	Vec<float> outOfBoundsAmount = imageEndCoordinate - Vec<float>(inputSize.dims-1);

	kernelEndIdx = kernelSize - Vec<size_t>(Vec<float>::max(1, outOfBoundsAmount +1));

	iterator = ImageDimensions(0,0,0);
	isEnd = false;
}

__device__ __host__ KernelIterator::~KernelIterator()
{
	isEnd = true;
	iterator.dims = Vec<size_t>(ULLONG_MAX);
	iterator.chan = INT_MAX;
	iterator.frame = INT_MAX;
	inputStartCoordinate = Vec<float>(FLT_MAX_EXP);
	kernelStartIdx = Vec<size_t>(ULLONG_MAX);
	kernelEndIdx = Vec<size_t>(0);
}

__device__ KernelIterator& KernelIterator::operator++()
{
	if(++iterator.dims.x>kernelEndIdx.x)
	{
		iterator.dims.x = kernelStartIdx.x;
		if(++iterator.dims.y>kernelEndIdx.y)
		{
			iterator.dims.y = kernelStartIdx.y;
			if(++iterator.dims.z>kernelEndIdx.z)
			{
				iterator.dims.z = kernelEndIdx.z;
				isChannelEnd = true;
				if(++iterator.chan>numChans)
				{
					iterator.chan = 0;
					isFrameEnd = true;
					if(++iterator.frame>numFrames)
					{
						iterator.frame = 0;
						isEnd = true;
					}
				}
			}
		}
	}

	return *this;
}

__device__ void KernelIterator::reset()
{
	isEnd = false;
	iterator = ImageDimensions(Vec<size_t>(0), 0, 0);
}

__device__ Vec<float> KernelIterator::getImageCoordinate() const
{
	return inputStartCoordinate + Vec<float>(iterator.dims);
}

__device__ unsigned int KernelIterator::getChannel() const
{
	return iterator.chan;
}

__device__ unsigned int KernelIterator::getFrame() const
{
	return iterator.frame;
}

__device__ size_t KernelIterator::getKernelIdx() const
{
	return iterator.dims.product();
}

__device__ ImageDimensions KernelIterator::getFullPos() const
{
	return iterator;
}


__device__ void GetThreadBlockCoordinate(Vec<size_t>& coordinate)
{
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;
}


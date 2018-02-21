#include "KernelIterator.cuh"

#include <limits.h>
#include <float.h>

__device__ KernelIterator::KernelIterator(Vec<size_t> inputPos, Vec<size_t> inputSize, Vec<size_t> kernelSize)
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
	Vec<float> outOfBoundsAmount = imageEndCoordinate - Vec<float>(inputSize-1);

	kernelEndIdx = kernelSize - Vec<size_t>(Vec<float>::max(1, outOfBoundsAmount +1));

	iterator = Vec<size_t>(0);
	isEnd = false;
}

__device__ __host__ KernelIterator::~KernelIterator()
{
	isEnd = true;
	iterator = Vec<size_t>(ULLONG_MAX);
	inputStartCoordinate = Vec<float>(FLT_MAX_EXP);
	kernelStartIdx = Vec<size_t>(ULLONG_MAX);
	kernelEndIdx = Vec<size_t>(0);
}

__device__ KernelIterator& KernelIterator::operator++()
{
	if(++iterator.x>kernelEndIdx.x)
	{
		iterator.x = kernelStartIdx.x;
		
		if(++iterator.y>kernelEndIdx.y)
		{
			iterator.y = kernelStartIdx.y;
			if(++iterator.z>kernelEndIdx.z)
			{
				iterator.z = kernelEndIdx.z;
				isEnd = true;
			}
		}
	}

	return *this;
}

__device__ Vec<float> KernelIterator::getImageCoordinate()
{
	return inputStartCoordinate + Vec<float>(iterator);
}

__device__ Vec<size_t> KernelIterator::getKernelIdx()
{
	return iterator;
}
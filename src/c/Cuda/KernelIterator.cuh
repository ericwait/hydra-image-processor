#pragma once

#include "Vec.h"
#include "ImageDimensions.cuh"

#include <cuda_runtime.h>

__device__ void GetThreadBlockCoordinate(Vec<size_t>& coordinate);

class KernelIterator
{
public:
	__device__ KernelIterator(Vec<size_t> inputPos, ImageDimensions inputSize, Vec<size_t> kernelSize);
	__device__ __host__ ~KernelIterator();

	__device__ KernelIterator& operator++();
	__device__ bool end() const { return isEnd; }
	__device__ bool channelEnd() const { return isChannelEnd; }
	__device__ bool frameEnd() const { return isFrameEnd; }

	__device__ void nextChannel() { isChannelEnd = false; }
	__device__ void nextFrame() { isFrameEnd = false; }
	__device__ void reset();

	__device__ Vec<float> getImageCoordinate() const;
	__device__ unsigned int getChannel() const;
	__device__ unsigned int getFrame() const;
	__device__ size_t getKernelIdx() const;
	__device__ ImageDimensions getFullPos() const;
	
private:
	__device__ __host__ KernelIterator() {}

	// This is the first coordinate that to be used in the image
	// If the coordinate is non-integer, then the image should be interpolated
	Vec<float> inputStartCoordinate;

	// This is the first index of the kernel to use
	Vec<size_t> kernelStartIdx;

	// This is the last index of the kernel to use (e.g. end+1 is out of bounds)
	Vec<size_t> kernelEndIdx;

	// Save the number of channels and frames
	unsigned int numChans;
	unsigned int numFrames;

	// This indicates the current position
	ImageDimensions iterator;

	bool isChannelEnd;
	bool isFrameEnd;
	bool isEnd;
};


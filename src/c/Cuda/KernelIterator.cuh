#pragma once

#include "Vec.h"
#include <cuda_runtime.h>

__device__ void GetThreadBlockCoordinate(Vec<std::size_t>& coordinate);

class KernelIterator
{
public:
	__device__ KernelIterator(Vec<std::size_t> inputPos, Vec<std::size_t> inputSize, Vec<std::size_t> kernelSize);
	__host__ __device__ ~KernelIterator();

	__device__ KernelIterator& operator++();
	__device__ bool end() { return isEnd; }
	__device__ void reset();
	__device__ Vec<float> getImageCoordinate();
	__device__ std::size_t getKernelIdx();
	__device__ Vec<std::size_t> getKernelCoordinate() { return iterator; }

private:
	__device__ __host__ KernelIterator() {}

	// This is the first coordinate that to be used in the image
	// If the coordinate is non-integer, then the image should be interpolated
	Vec<float> inputStartCoordinate;

	// This is the first index of the kernel to use
	Vec<std::size_t> kernelStartIdx;

	// This is the last index of the kernel to use (e.g. end+1 is out of bounds)
	Vec<std::size_t> kernelEndIdx;

	// This indicates the current position
	Vec<std::size_t> iterator;

	Vec<std::size_t> dims;

	bool isEnd;
};

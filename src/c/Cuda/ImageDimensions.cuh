#pragma once
#include "Vec.h"
#include "cuda_runtime.h"

class ImageDimensions
{
public:
	__host__ __device__ ImageDimensions()
	{
		dims = Vec<size_t>(0);
		chan = 0;
		frame = 0;
	}

	__host__ __device__ ImageDimensions(Vec<size_t> spatialDimensionsIn, unsigned int numChannelsIn, unsigned int numFramesIn)
	{
		dims = spatialDimensionsIn;
		chan = numChannelsIn;
		frame = numFramesIn;
	}

	__host__ __device__ ImageDimensions(size_t spatialDimensionsIn, unsigned int numChannelsIn, unsigned int numFramesIn)
	{
		dims = Vec<size_t>(spatialDimensionsIn);
		chan = numChannelsIn;
		frame = numFramesIn;
	}

	__host__ __device__ ImageDimensions(const ImageDimensions& other)
	{
		dims = other.dims;
		chan = other.chan;
		frame = other.frame;
	}

	__host__ __device__ ImageDimensions& operator=(const ImageDimensions& other)
	{
		dims = other.dims;
		chan = other.chan;
		frame = other.frame;

		return *this;
	}

	__host__ __device__ ImageDimensions operator+(ImageDimensions adder) const
	{
		ImageDimensions outDims;
		outDims.dims = this->dims+adder.dims;
		outDims.chan = this->chan+adder.chan;
		outDims.frame = this->frame+adder.frame;

		return outDims;
	}

	__host__ __device__ bool operator>=(const ImageDimensions& other) const
	{
		return dims>=other.dims && chan>=other.chan && frame>=other.frame;
	}

	__host__ __device__ bool operator!=(const ImageDimensions& other) const
	{
		return dims!=other.dims && chan!=other.chan && frame!=other.frame;
	}

	__host__ __device__ bool operator==(const ImageDimensions& other) const
	{
		return dims==other.dims && chan==other.chan && frame==other.frame;
	}
	__host__ __device__ size_t linearAddressAt(ImageDimensions coordinate) const
	{
		size_t index =
			coordinate.dims.x +
			coordinate.dims.y * dims.x +
			coordinate.dims.z * dims.x * dims.y +
			coordinate.chan	  * dims.x * dims.y * dims.z +
			coordinate.frame  * dims.x * dims.y * dims.z * chan;

		return index;
	}

	__host__ __device__ ImageDimensions coordAddressOf(size_t index) const
	{
		ImageDimensions coordOut;
		size_t stride = dims.product()*chan;
		coordOut.frame = index/stride;

		index -= coordOut.frame * stride;
		stride /= chan;
		coordOut.chan = index/stride;

		index -= coordOut.chan * stride;
		stride /= dims.z;
		coordOut.dims.z = index/stride;

		index -= coordOut.dims.z * stride;
		stride /= dims.y;
		coordOut.dims.y = index/stride;

		index -= coordOut.dims.y * stride;
		stride /= dims.x;
		coordOut.dims.x = index/stride;

		return coordOut;
	}

	__host__ __device__ size_t getChanStride() const 
	{
		return dims.product(); 
	}

	__host__ __device__ size_t getFrameStride() const
	{
		return getChanStride() * chan;
	}

	__host__ __device__ size_t getNumElements() const
	{
		return getFrameStride() * frame;
	}

	Vec<size_t> dims;
	unsigned int chan;
	unsigned int frame;
};

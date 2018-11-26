#pragma once
#include "Vec.h"

#include <type_traits>

#ifdef __CUDACC__
#define MIXED_PREFIX __host__ __device__
#else
#define MIXED_PREFIX
#endif

class ImageDimensions
{
public:
	MIXED_PREFIX ImageDimensions()
	{
		dims = Vec<std::size_t>(0);
		chan = 0;
		frame = 0;
	}

	MIXED_PREFIX ImageDimensions(Vec<std::size_t> spatialDimensionsIn, unsigned int numChannelsIn, unsigned int numFramesIn)
	{
		dims = spatialDimensionsIn;
		chan = numChannelsIn;
		frame = numFramesIn;
	}

	MIXED_PREFIX ImageDimensions(std::size_t spatialDimensionsIn, unsigned int numChannelsIn, unsigned int numFramesIn)
	{
		dims = Vec<std::size_t>(spatialDimensionsIn);
		chan = numChannelsIn;
		frame = numFramesIn;
	}

	MIXED_PREFIX ImageDimensions(const ImageDimensions& other)
	{
		dims = other.dims;
		chan = other.chan;
		frame = other.frame;
	}

	MIXED_PREFIX ImageDimensions& operator=(const ImageDimensions& other)
	{
		dims = other.dims;
		chan = other.chan;
		frame = other.frame;

		return *this;
	}

	MIXED_PREFIX ImageDimensions operator+(ImageDimensions adder)
	{
		ImageDimensions outDims;
		outDims.dims = this->dims+adder.dims;
		outDims.chan = this->chan+adder.chan;
		outDims.frame = this->frame+adder.frame;

		return outDims;
	}

	MIXED_PREFIX ImageDimensions operator-(ImageDimensions adder)
	{
		ImageDimensions outDims;
		outDims.dims = this->dims - adder.dims;
		outDims.chan = this->chan - adder.chan;
		outDims.frame = this->frame - adder.frame;

		return outDims;
	}

	MIXED_PREFIX ImageDimensions& operator+=(ImageDimensions adder)
	{
		this->dims = this->dims + adder.dims;
		this->chan = this->chan + adder.chan;
		this->frame = this->frame + adder.frame;

		return *this;
	}

	MIXED_PREFIX ImageDimensions& operator+=(Vec<std::size_t> adder)
	{
		this->dims = this->dims + adder;

		return *this;
	}

	MIXED_PREFIX bool operator>=(const ImageDimensions& other) const
	{
		return dims>=other.dims && chan>=other.chan && frame>=other.frame;
	}

	MIXED_PREFIX bool operator!=(const ImageDimensions& other) const
	{
		return dims!=other.dims && chan!=other.chan && frame!=other.frame;
	}

	MIXED_PREFIX bool operator==(const ImageDimensions& other) const
	{
		return dims==other.dims && chan==other.chan && frame==other.frame;
	}
	MIXED_PREFIX std::size_t linearAddressAt(ImageDimensions coordinate) const
	{
		std::size_t index =
			coordinate.dims.x +
			coordinate.dims.y * dims.x +
			coordinate.dims.z * dims.x * dims.y +
			coordinate.chan	  * dims.x * dims.y * dims.z +
			coordinate.frame  * dims.x * dims.y * dims.z * chan;

		return index;
	}

	MIXED_PREFIX ImageDimensions coordAddressOf(std::size_t index) const
	{
		ImageDimensions coordOut;
		std::size_t stride = dims.product()*(std::size_t)chan;
		coordOut.frame = (unsigned int)(index/stride);

		index -= coordOut.frame * stride;
		stride /= chan;
		coordOut.chan = (unsigned int)(index/stride);

		index -= coordOut.chan * stride;
		stride /= dims.z;
		coordOut.dims.z = (unsigned int)(index/stride);

		index -= coordOut.dims.z * stride;
		stride /= dims.y;
		coordOut.dims.y = (unsigned int)(index/stride);

		index -= coordOut.dims.y * stride;
		stride /= dims.x;
		coordOut.dims.x = (unsigned int)(index/stride);

		return coordOut;
	}

	MIXED_PREFIX std::size_t getChanStride() const 
	{
		return dims.product(); 
	}

	MIXED_PREFIX std::size_t getFrameStride() const
	{
		return getChanStride() * chan;
	}

	MIXED_PREFIX std::size_t getNumElements() const
	{
		return getFrameStride() * frame;
	}

	Vec<std::size_t> dims;
	unsigned int chan;
	unsigned int frame;
};

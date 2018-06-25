#pragma once
#include "ImageDimensions.cuh"

#include <cassert>

template<class PixelType>
class ImageContainer
{
public:
	ImageContainer()
	{
		reset();
	}

	ImageContainer(PixelType val, Vec<size_t> dimsIn, unsigned char nChannels = 1, unsigned int nFrames = 1)
	{
		reset();
		image = new PixelType[dimsIn.product()*nChannels*nFrames];
		dimensions.dims = dimsIn;
		dimensions.chan = nChannels;
		dimensions.frame = nFrames;
	}

	ImageContainer(PixelType* imagePtr, Vec<size_t> dimsIn, unsigned char nChannels = 1, unsigned int nFrames = 1)
	{
		image = imagePtr;
		dimensions.dims = dimsIn;
		dimensions.chan = nChannels;
		dimensions.frame = nFrames;
	}

	ImageContainer(PixelType* imagePtr, ImageDimensions dimsIn)
	{
		image = imagePtr;
		dimensions = dimsIn;
	}

	ImageContainer(const ImageContainer& other)
	{
		image = other.getPtr();
		dimensions = other.getDims();
	}

	~ImageContainer()
	{
		reset();
	}

	void setPointer(PixelType* imagePtr, ImageDimensions dimsIn)
	{
		assert(image == NULL);
		image = imagePtr;
		dimensions = dimsIn;
	}

	void resize(ImageDimensions dims)
	{
		assert(image == NULL);
		image = new PixelType[dims.getNumElements()];
		dimensions = dims;
	}

	const PixelType* getConstPtr() const { return image; }
	PixelType* getPtr() const { return image; }
	ImageDimensions getDims() const { return dimensions; }
	Vec<size_t> getSpatialDims() const { return dimensions.dims; }
	unsigned int getNumChannels() const { return dimensions.chan; }
	unsigned int getNumFrames() const { return dimensions.frame; }
	size_t getNumElements() const { return dimensions.getNumElements(); }

	void clear() { delete[] image; dimensions = ImageDimensions(); }
	
private:
	void reset()
	{
		image = NULL;
		dimensions = ImageDimensions();
	}

	PixelType* image;
	ImageDimensions dimensions;
};

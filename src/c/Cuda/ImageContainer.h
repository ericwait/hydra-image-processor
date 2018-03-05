#pragma once
#include "ImageDimensions.cuh"

template<class PixelType>
class ImageContainer
{
public:
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

	~ImageContainer()
	{
		image = NULL;
		dimensions = ImageDimensions();
	}

	const PixelType* getConstPtr() { return image; }
	PixelType* getPtr() { return image; }
	ImageDimensions getDims() const { return dimensions; }
	Vec<size_t> getSpatialDims() const { return dimensions.dims; }
	unsigned int getNumChannels() const { return dimensions.chan; }
	unsigned int getNumFrames() const { return dimensions.frame; }
	
private:
	ImageContainer();

	PixelType* image;
	ImageDimensions dimensions;
};

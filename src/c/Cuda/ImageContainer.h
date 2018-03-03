#pragma once

struct ImageDimensions
{
	Vec<size_t> spatialDimensions = Vec<size_t>(0);
	unsigned char numChannels = 0;
	unsigned int numFrames = 0;
};

template<class PixelType>
class ImageContainer
{
public:
	ImageContainer(PixelType* imagePtr, Vec<size_t> dims, unsigned char nChannels = 1, unsigned int nFrames = 1)
	{
		image = imagePtr;
		dims.spatialDimensions = dims;
		dims.numChannels = nChannels;
		dims.numFrames = nFrames;
	}

	~ImageContainer()
	{
		image = NULL;
		dims = ImageDimensions;
	}

	const PixelType* getConstPtr() { return image; }
	PixelType* getPtr() { return image; }
	Vec<size_t> getDims() const { return dims.spatialDimensions; }
	unsigned char getNumChannels() const { return dims.numChannels; }
	unsigned int getNumFrames() const { return dims.numFrames; }
	
private:
	ImageContainer();

	PixelType* image;
	ImageDimensions dims;
};

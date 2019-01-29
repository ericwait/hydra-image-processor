#pragma once
#include "ImageDimensions.cuh"

#include <cassert>
#include <cstddef>
#include <memory>

template <class PixelType>
class ImageOwner;

//////////
// ImageView - Non-owning image wrapper, supports default copy and move semantics.
//   Can also be implicitly created from an ImageOwner.
template<class PixelType>
class ImageView
{
public:
	// Empty image view
	ImageView(): imageView(nullptr), dimensions() {}

	// Implicitly ImageView from an owning container object
	ImageView(const ImageOwner<PixelType>& owner)
		: imageView(owner.getPtr()), dimensions(owner.getDims())
	{}

	// Ability to assign to assign an owning container to an ImageView
	ImageView& operator=(const ImageOwner<PixelType>& owner)
	{
		imageView = owner.getPtr();
		dimensions = owner.getDims();

		return (*this);
	}


	ImageView(PixelType* imagePtr, Vec<std::size_t> dimsIn, unsigned char nChannels = 1, unsigned int nFrames = 1)
		: imageView(imagePtr), dimensions(dimsIn, nChannels, nFrames)
	{}

	ImageView(PixelType* imagePtr, ImageDimensions dimsIn)
		: ImageView(imagePtr, dimsIn.dims, dimsIn.chan, dimsIn.frame)
	{}


	// Member access functions
	PixelType* getPtr() const { return imageView; }
	const PixelType* getConstPtr() const { return imageView; }

	ImageDimensions getDims() const { return dimensions; }
	Vec<std::size_t> getSpatialDims() const { return dimensions.dims; }

	unsigned int getNumChannels() const { return dimensions.chan; }
	unsigned int getNumFrames() const { return dimensions.frame; }

	std::size_t getNumElements() const { return dimensions.getNumElements(); }

private:
	ImageDimensions dimensions;
	PixelType* imageView;
};


//////////
// ImageOwner - Owns the underlying image pointer and will clean up
//   when object goes out of scope. Supports only move-semantics (like unique_ptr)
template<class PixelType>
class ImageOwner
{
public:
	// Disable copy operations
	ImageOwner(const ImageOwner&) = delete;
	ImageOwner& operator=(const ImageOwner&) = delete;

	// Enable default move semantics (works with trivial types and unique_ptr)
	ImageOwner(ImageOwner&&) = default;
	ImageOwner& operator=(ImageOwner&&) = default;

	// Constructor for an empty image
	ImageOwner(): image(nullptr), dimensions() {}

	// Construct an image with given dimensions
	ImageOwner(const Vec<std::size_t>& dimsIn, unsigned char nChannels = 1, unsigned int nFrames = 1)
		: dimensions(dimsIn, nChannels, nFrames)
	{
		std::size_t numEl = dimensions.getNumElements();
		image = std::unique_ptr<PixelType[]>(new PixelType[numEl]);
	}

	ImageOwner(const ImageDimensions& dimsIn)
		: ImageOwner(dimsIn.dims, dimsIn.chan, dimsIn.frame)
	{}


	// Create image with uniform values
	ImageOwner(PixelType val, const Vec<std::size_t>& dimsIn, unsigned char nChannels = 1, unsigned int nFrames = 1)
		: ImageOwner(dimsIn, nChannels, nFrames)
	{
		std::size_t numEl = dimensions.getNumElements();
		for ( std::size_t i = 0; i < numEl; ++i )
			image[i] = val;
	}

	ImageOwner(PixelType val, const ImageDimensions& dimsIn)
		: ImageOwner(val, dimsIn.dims, dimsIn.chan, dimsIn.frame)
	{}


	// Member access functions
	PixelType* getPtr() const { return image.get(); }
	const PixelType* getConstPtr() const { return image.get(); }

	ImageDimensions getDims() const { return dimensions; }
	Vec<std::size_t> getSpatialDims() const { return dimensions.dims; }

	unsigned int getNumChannels() const { return dimensions.chan; }
	unsigned int getNumFrames() const { return dimensions.frame; }

	std::size_t getNumElements() const { return dimensions.getNumElements(); }

private:
	ImageDimensions dimensions;
	std::unique_ptr<PixelType[]> image;
};

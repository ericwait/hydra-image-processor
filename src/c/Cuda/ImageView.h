/**
 * @file ImageView.h
 * @brief Image container classes for managing pixel data
 *
 * Provides two image container classes:
 * - ImageView: Non-owning view of image data (lightweight, copyable)
 * - ImageOwner: Owning container that manages image memory (move-only)
 */

#pragma once
#include "ImageDimensions.cuh"

#include <cassert>
#include <cstddef>
#include <memory>

template <class PixelType>
class ImageOwner;

/**
 * @brief Non-owning image wrapper that provides a view into image data
 *
 * ImageView is a lightweight, copyable class that references image data
 * without owning it. It can be implicitly created from an ImageOwner and
 * supports default copy and move semantics. Use this when you need to
 * pass image data without transferring ownership.
 *
 * @tparam PixelType The data type of the image pixels (e.g., float, unsigned char)
 */
template<class PixelType>
class ImageView
{
public:
	/**
	 * @brief Default constructor - creates an empty image view
	 */
	ImageView(): dimensions(), imageView(nullptr) {}

	/**
	 * @brief Implicit conversion from ImageOwner
	 *
	 * Creates a view of an owning image container without transferring ownership.
	 *
	 * @param owner The ImageOwner to create a view from
	 */
	ImageView(const ImageOwner<PixelType>& owner)
		: dimensions(owner.getDims()), imageView(owner.getPtr())
	{}

	/**
	 * @brief Constructs an image view from existing data
	 *
	 * @param imagePtr Pointer to the image data
	 * @param dimsIn Spatial dimensions (x, y, z)
	 * @param nChannels Number of color channels (default: 1)
	 * @param nFrames Number of time frames (default: 1)
	 */
	ImageView(PixelType* imagePtr, Vec<std::size_t> dimsIn, unsigned char nChannels = 1, unsigned int nFrames = 1)
		: dimensions(dimsIn, nChannels, nFrames), imageView(imagePtr)
	{}

	/**
	 * @brief Constructs an image view from existing data with ImageDimensions
	 *
	 * @param imagePtr Pointer to the image data
	 * @param dimsIn Complete image dimensions including channels and frames
	 */
	ImageView(PixelType* imagePtr, ImageDimensions dimsIn)
		: ImageView(imagePtr, dimsIn.dims, dimsIn.chan, dimsIn.frame)
	{}

	/**
	 * @brief Gets mutable pointer to image data
	 * @return Pointer to the image pixel data
	 */
	PixelType* getPtr() const { return imageView; }

	/**
	 * @brief Gets const pointer to image data
	 * @return Const pointer to the image pixel data
	 */
	const PixelType* getConstPtr() const { return imageView; }

	/**
	 * @brief Gets complete image dimensions
	 * @return ImageDimensions structure containing spatial dims, channels, and frames
	 */
	ImageDimensions getDims() const { return dimensions; }

	/**
	 * @brief Gets spatial dimensions only
	 * @return 3D vector containing x, y, z dimensions
	 */
	Vec<std::size_t> getSpatialDims() const { return dimensions.dims; }

	/**
	 * @brief Gets number of color channels
	 * @return Number of channels in the image
	 */
	unsigned int getNumChannels() const { return dimensions.chan; }

	/**
	 * @brief Gets number of time frames
	 * @return Number of frames in the image
	 */
	unsigned int getNumFrames() const { return dimensions.frame; }

	/**
	 * @brief Gets total number of elements in the image
	 * @return Total number of pixels (width * height * depth * channels * frames)
	 */
	std::size_t getNumElements() const { return dimensions.getNumElements(); }

private:
	ImageDimensions dimensions;
	PixelType* imageView;
};


/**
 * @brief Owning image container that manages image memory
 *
 * ImageOwner owns the underlying image data and automatically cleans up
 * when the object goes out of scope. Supports only move semantics (similar
 * to std::unique_ptr) to prevent accidental copies of large image data.
 * Can be implicitly converted to ImageView for passing to functions.
 *
 * @tparam PixelType The data type of the image pixels (e.g., float, unsigned char)
 */
template<class PixelType>
class ImageOwner
{
public:
	/// @brief Disable copy constructor
	ImageOwner(const ImageOwner&) = delete;
	/// @brief Disable copy assignment
	ImageOwner& operator=(const ImageOwner&) = delete;

	/// @brief Enable default move constructor
	ImageOwner(ImageOwner&&) = default;
	/// @brief Enable default move assignment
	ImageOwner& operator=(ImageOwner&&) = default;

	/**
	 * @brief Default constructor - creates an empty image
	 */
	ImageOwner(): dimensions(), image(nullptr) {}

	/**
	 * @brief Constructs an image with the given dimensions
	 *
	 * Allocates memory for an image with the specified dimensions. Pixel values
	 * are uninitialized.
	 *
	 * @param dimsIn Spatial dimensions (x, y, z)
	 * @param nChannels Number of color channels (default: 1)
	 * @param nFrames Number of time frames (default: 1)
	 */
	ImageOwner(const Vec<std::size_t>& dimsIn, unsigned char nChannels = 1, unsigned int nFrames = 1)
		: dimensions(dimsIn, nChannels, nFrames)
	{
		std::size_t numEl = dimensions.getNumElements();
		image = std::unique_ptr<PixelType[]>(new PixelType[numEl]);
	}

	/**
	 * @brief Constructs an image with ImageDimensions
	 *
	 * @param dimsIn Complete image dimensions including channels and frames
	 */
	ImageOwner(const ImageDimensions& dimsIn)
		: ImageOwner(dimsIn.dims, dimsIn.chan, dimsIn.frame)
	{}

	/**
	 * @brief Constructs an image initialized with a uniform value
	 *
	 * Allocates and initializes all pixels to the specified value.
	 *
	 * @param val The value to initialize all pixels to
	 * @param dimsIn Spatial dimensions (x, y, z)
	 * @param nChannels Number of color channels (default: 1)
	 * @param nFrames Number of time frames (default: 1)
	 */
	ImageOwner(PixelType val, const Vec<std::size_t>& dimsIn, unsigned char nChannels = 1, unsigned int nFrames = 1)
		: ImageOwner(dimsIn, nChannels, nFrames)
	{
		std::size_t numEl = dimensions.getNumElements();
		for ( std::size_t i = 0; i < numEl; ++i )
			image[i] = val;
	}

	/**
	 * @brief Constructs an image initialized with a uniform value using ImageDimensions
	 *
	 * @param val The value to initialize all pixels to
	 * @param dimsIn Complete image dimensions including channels and frames
	 */
	ImageOwner(PixelType val, const ImageDimensions& dimsIn)
		: ImageOwner(val, dimsIn.dims, dimsIn.chan, dimsIn.frame)
	{}

	/**
	 * @brief Gets mutable pointer to image data
	 * @return Pointer to the owned image pixel data
	 */
	PixelType* getPtr() const { return image.get(); }

	/**
	 * @brief Gets const pointer to image data
	 * @return Const pointer to the owned image pixel data
	 */
	const PixelType* getConstPtr() const { return image.get(); }

	/**
	 * @brief Gets complete image dimensions
	 * @return ImageDimensions structure containing spatial dims, channels, and frames
	 */
	ImageDimensions getDims() const { return dimensions; }

	/**
	 * @brief Gets spatial dimensions only
	 * @return 3D vector containing x, y, z dimensions
	 */
	Vec<std::size_t> getSpatialDims() const { return dimensions.dims; }

	/**
	 * @brief Gets number of color channels
	 * @return Number of channels in the image
	 */
	unsigned int getNumChannels() const { return dimensions.chan; }

	/**
	 * @brief Gets number of time frames
	 * @return Number of frames in the image
	 */
	unsigned int getNumFrames() const { return dimensions.frame; }

	/**
	 * @brief Gets total number of elements in the image
	 * @return Total number of pixels (width * height * depth * channels * frames)
	 */
	std::size_t getNumElements() const { return dimensions.getNumElements(); }

private:
	ImageDimensions dimensions;
	std::unique_ptr<PixelType[]> image;
};

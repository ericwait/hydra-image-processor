#pragma once

#include "Vec.h"
#include <string>

typedef unsigned char HostPixelType;

class ImageContainer
{
public:
	ImageContainer(size_t width, size_t height, size_t depth, bool isColumnMajor=false);
	ImageContainer(Vec<size_t> dims, bool isColumnMajor=false);
	ImageContainer(HostPixelType* imageIn, Vec<size_t> dims, bool isColumnMajor=false);
	ImageContainer(const ImageContainer& image){copy(image);}
	~ImageContainer(){clear();}
	ImageContainer& operator=(const ImageContainer& image){copy(image); return *this;}

	HostPixelType& operator[](Vec<size_t> coordinate);
	const HostPixelType& operator[](Vec<size_t> coordinate) const;
	HostPixelType& at(Vec<size_t> coordinate);
	const HostPixelType& at(Vec<size_t> coordinate) const;
	HostPixelType getPixelValue(size_t x, size_t y, size_t z) const;
	HostPixelType getPixelValue(Vec<size_t> coordinate) const;
	const HostPixelType* getConstMemoryPointer() const {return image;}
	const HostPixelType* ImageContainer::getConstROIData (size_t minX, size_t sizeX, size_t minY,
		size_t sizeY, size_t minZ, size_t sizeZ) const;

	const HostPixelType* getConstROIData(Vec<size_t> startIndex, Vec<size_t> size) const;
	HostPixelType* getMemoryPointer(){return image;}
	Vec<size_t> getDims() const {return imageDims;}
	size_t getWidth() const {return imageDims.x;}
	size_t getHeight() const {return imageDims.y;}
	size_t getDepth() const {return imageDims.z;}

	void setROIData(HostPixelType* image, Vec<size_t> startIndex, Vec<size_t> size);
	void setPixelValue(size_t x, size_t y, size_t z, unsigned char val);
	void setPixelValue(Vec<size_t> coordinate, HostPixelType val);
	void loadImage(const HostPixelType* imageIn, size_t width, size_t height, size_t depth, bool isColumnMajor=false);
	void loadImage(const HostPixelType* imageIn, Vec<size_t> dims, bool isColumnMajor=false);

private:
	ImageContainer();
	void copy(const ImageContainer& image);
	void clear();
	void defaults() 
	{
		imageDims = Vec<size_t>(0,0,0);
		columnMajor = false;
		image = NULL;
	}

	Vec<size_t> imageDims;
	bool columnMajor;
	HostPixelType*	image;
};

#pragma once

#include "Vec.h"
#include <string>

typedef unsigned char HostPixelType;

class ImageContainer
{
public:
	ImageContainer(unsigned int width, unsigned int height, unsigned int depth, bool isColumnMajor=false);
	ImageContainer(Vec<unsigned int> dims, bool isColumnMajor=false);
	ImageContainer(HostPixelType* imageIn, Vec<unsigned int> dims, bool isColumnMajor=false);
	ImageContainer(const ImageContainer& image){copy(image);}
	~ImageContainer(){clear();}
	ImageContainer& operator=(const ImageContainer& image){copy(image); return *this;}

	HostPixelType& operator[](Vec<unsigned int> coordinate);
	const HostPixelType& operator[](Vec<unsigned int> coordinate) const;
	HostPixelType& at(Vec<unsigned int> coordinate);
	const HostPixelType& at(Vec<unsigned int> coordinate) const;
	HostPixelType getPixelValue(unsigned int x, unsigned int y, unsigned int z) const;
	HostPixelType getPixelValue(Vec<unsigned int> coordinate) const;
	const HostPixelType* getConstMemoryPointer() const {return image;}
	const HostPixelType* ImageContainer::getConstROIData (unsigned int minX, unsigned int sizeX, unsigned int minY,
		unsigned int sizeY, unsigned int minZ, unsigned int sizeZ) const;

	const HostPixelType* getConstROIData(Vec<unsigned int> startIndex, Vec<unsigned int> size) const;
	HostPixelType* getMemoryPointer(){return image;}
	Vec<unsigned int> getDims() const {return imageDims;}
	unsigned int getWidth() const {return imageDims.x;}
	unsigned int getHeight() const {return imageDims.y;}
	unsigned int getDepth() const {return imageDims.z;}

	void setROIData(HostPixelType* image, Vec<unsigned int> startIndex, Vec<unsigned int> size);
	void setPixelValue(unsigned int x, unsigned int y, unsigned int z, unsigned char val);
	void setPixelValue(Vec<unsigned int> coordinate, HostPixelType val);
	void loadImage(const HostPixelType* imageIn, unsigned int width, unsigned int height, unsigned int depth, bool isColumnMajor=false);
	void loadImage(const HostPixelType* imageIn, Vec<unsigned int> dims, bool isColumnMajor=false);

private:
	ImageContainer();
	void copy(const ImageContainer& image);
	void clear();
	void defaults() 
	{
		imageDims = Vec<unsigned int>(0,0,0);
		columnMajor = false;
		image = NULL;
	}

	Vec<unsigned int> imageDims;
	bool columnMajor;
	HostPixelType*	image;
};

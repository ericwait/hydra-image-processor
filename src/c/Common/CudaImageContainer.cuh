#pragma once

#define DEVICE_VEC
#include "Vec.h"
#include <string>

typedef unsigned char DevicePixelType;

class CudaImageContainer
{
public:
	CudaImageContainer(size_t width, size_t height, size_t depth, bool isColumnMajor=false);
	CudaImageContainer(DeviceVec<size_t> dims, bool isColumnMajor=false);
	CudaImageContainer(DevicePixelType* imageIn, DeviceVec<size_t> dims, bool isColumnMajor=false);
	CudaImageContainer(const CudaImageContainer& image){copy(image);}
	~CudaImageContainer(){clear();}
	CudaImageContainer& operator=(const CudaImageContainer& image){copy(image); return *this;}

	DevicePixelType& operator[](DeviceVec<size_t> coordinate);
	const DevicePixelType& operator[](DeviceVec<size_t> coordinate) const;
	DevicePixelType& at(DeviceVec<size_t> coordinate);
	const DevicePixelType& at(DeviceVec<size_t> coordinate) const;
	DevicePixelType getPixelValue(size_t x, size_t y, size_t z) const;
	DevicePixelType getPixelValue(DeviceVec<size_t> coordinate) const;
	const DevicePixelType* getConstMemoryPointer() const {return image;}
	const DevicePixelType* CudaImageContainer::getConstROIData (size_t minX, size_t sizeX, size_t minY,
		size_t sizeY, size_t minZ, size_t sizeZ) const;

	const DevicePixelType* getConstROIData(DeviceVec<size_t> startIndex, DeviceVec<size_t> size) const;
	DevicePixelType* getMemoryPointer(){return image;}
	DeviceVec<size_t> getDims() const {return imageDims;}
	size_t getWidth() const {return imageDims.x;}
	size_t getHeight() const {return imageDims.y;}
	size_t getDepth() const {return imageDims.z;}

	void setROIData(DevicePixelType* image, DeviceVec<size_t> startIndex, DeviceVec<size_t> size);
	void setPixelValue(size_t x, size_t y, size_t z, unsigned char val);
	void setPixelValue(DeviceVec<size_t> coordinate, DevicePixelType val);
	void loadImage(const DevicePixelType* imageIn, size_t width, size_t height, size_t depth, bool isColumnMajor=false);
	void loadImage(const DevicePixelType* imageIn, DeviceVec<size_t> dims, bool isColumnMajor=false);

private:
	CudaImageContainer();
	void copy(const CudaImageContainer& image);
	void clear();
	void defaults() 
	{
		imageDims = DeviceVec<size_t>(0,0,0);
		columnMajor = false;
		image = NULL;
	}

	DeviceVec<size_t> imageDims;
	bool columnMajor;
	DevicePixelType*	image;
};
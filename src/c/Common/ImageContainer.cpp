#include "ImageContainer.h"

ImageContainer::ImageContainer(size_t width, size_t height, size_t depth, bool isColumnMajor/*=false*/)
{
	defaults();
	imageDims.x = width;
	imageDims.y = height;
	imageDims.z = depth;

	columnMajor = isColumnMajor;

	image = new HostPixelType[imageDims.product()];
}

ImageContainer::ImageContainer(Vec<size_t> dimsIn, bool isColumnMajor/*=false*/)
{
	defaults();
	imageDims = dimsIn;

	columnMajor = isColumnMajor;

	image = new HostPixelType[imageDims.product()];
}

ImageContainer::ImageContainer( HostPixelType* imageIn, Vec<size_t> dims, bool isColumnMajor/*=false*/)
{
	defaults();
	columnMajor = isColumnMajor;
	loadImage(imageIn,dims,isColumnMajor);
}

void ImageContainer::copy(const ImageContainer& im)
{
	clear();

	image = new HostPixelType[imageDims.product()];
	memcpy((void*)image,(void*)(im.getConstMemoryPointer()),sizeof(HostPixelType)*imageDims.product());
}

void ImageContainer::clear()
{
	defaults();

	if (image)
	{
		delete[] image;
		image = NULL;
	}
}

HostPixelType ImageContainer::getPixelValue(size_t x, size_t y, size_t z) const
{
	return getPixelValue(Vec<size_t>(x,y,z));
}

HostPixelType ImageContainer::getPixelValue(Vec<size_t> coordinate) const
{
	return image[imageDims.linearAddressAt(coordinate)];
}

void ImageContainer::setPixelValue(size_t x, size_t y, size_t z, unsigned char val)
{
	setPixelValue(Vec<size_t>(x,y,z),val);
}

void ImageContainer::setPixelValue(Vec<size_t> coordinate, unsigned char val)
{
	image[imageDims.linearAddressAt(coordinate)] = val;
}

const HostPixelType* ImageContainer::getConstROIData (size_t minX, size_t sizeX, size_t minY,
													  size_t sizeY, size_t minZ, size_t sizeZ) const
{
	return getConstROIData(Vec<size_t>(minX,minY,minZ), Vec<size_t>(sizeX,sizeY,sizeZ));
}

const HostPixelType* ImageContainer::getConstROIData(Vec<size_t> startIndex, Vec<size_t> size) const
{
	HostPixelType* imageOut = new HostPixelType[size.product()];

	size_t i=0;
	Vec<size_t> curIdx(startIndex);
	for (curIdx.z=startIndex.z; curIdx.z-startIndex.z<size.z && curIdx.z<imageDims.z; ++curIdx.z)
	{
		for (curIdx.y=startIndex.y; curIdx.y-startIndex.y<size.y && curIdx.y<imageDims.y; ++curIdx.y)
		{
			for (curIdx.x=startIndex.x; curIdx.x-startIndex.x<size.x && curIdx.x<imageDims.x; ++curIdx.x)		
			{
				imageOut[i] = getPixelValue(curIdx);
				++i;
			}
		}
	}

	return imageOut;
}

void ImageContainer::loadImage( const HostPixelType* imageIn, Vec<size_t> dims, bool isColumnMajor/*=false*/ )
{
	if (!isColumnMajor)
	{
		if (dims!=imageDims)
		{
			if (image!=NULL)
			{
				delete[] image;
			}
			image = new HostPixelType[dims.product()];
			imageDims = dims;
		}

		memcpy(image,imageIn,sizeof(HostPixelType)*imageDims.product());
	}
	else
	{
// 		if (dims.x!=imageDims.y || dims.y!=imageDims.x || dims.z!=imageDims.z)
// 		{
		if (dims!=imageDims)
		{
			if (image!=NULL)
			{
				delete[] image;
			}
			image = new HostPixelType[dims.product()];
			imageDims = dims;
		}
		//TODO: take this out when the cuda storage can take column major buffers
		size_t i = 0;
		double acum = 0.0;
		int mx = -1;
		Vec<size_t> curInIdx(0,0,0);
		for (curInIdx.z=0; curInIdx.z<dims.z; ++curInIdx.z)
		{
			for (curInIdx.y=0; curInIdx.y<dims.y; ++curInIdx.y)
			{
				for (curInIdx.x=0; curInIdx.x<dims.x; ++curInIdx.x)
				{
					if (i>=imageDims.product())
						break;

					acum += image[i] = imageIn[imageDims.linearAddressAt(curInIdx,true)];

					if (image[i]>mx)
						mx = image[i];

					++i;
				}
			}
		}
		double mean = acum/i;
		columnMajor = false;
	}
}

void ImageContainer::loadImage( const HostPixelType* imageIn, size_t width, size_t height, size_t depth, bool isColumnMajor/*=false*/ )
{
	loadImage(imageIn,Vec<size_t>(width,height,depth),isColumnMajor);
}

HostPixelType& ImageContainer::operator[]( Vec<size_t> coordinate )
{
	return image[imageDims.linearAddressAt(coordinate,columnMajor)];
}

const HostPixelType& ImageContainer::operator[]( Vec<size_t> coordinate ) const
{
	return image[imageDims.linearAddressAt(coordinate,columnMajor)];
}

HostPixelType& ImageContainer::at( Vec<size_t> coordinate )
{
	return image[imageDims.linearAddressAt(coordinate,columnMajor)];
}

void ImageContainer::setROIData( HostPixelType* imageIn, Vec<size_t> startIndex, Vec<size_t> sizeIn )
{
	Vec<size_t> curIdx(0,0,0);
	for (curIdx.z=0; curIdx.z<sizeIn.z && curIdx.z+startIndex.z<imageDims.z; ++curIdx.z)
	{
		for (curIdx.y=0; curIdx.y<sizeIn.y && curIdx.y+startIndex.y<imageDims.y; ++curIdx.y)
		{
			for (curIdx.x=0; curIdx.x<sizeIn.x && curIdx.x+startIndex.x<imageDims.x; ++curIdx.x)		
			{
				setPixelValue(curIdx+startIndex,imageIn[sizeIn.linearAddressAt(curIdx)]);
			}
		}
	}
}

#include "ImageContainer.h"

ImageContainer::ImageContainer(unsigned int width, unsigned int height, unsigned int depth, bool isColumnMajor/*=false*/)
{
	defaults();
	imageDims.x = width;
	imageDims.y = height;
	imageDims.z = depth;

	columnMajor = isColumnMajor;

	image = new HostPixelType[imageDims.product()];
}

ImageContainer::ImageContainer(Vec<unsigned int> dimsIn, bool isColumnMajor/*=false*/)
{
	defaults();
	imageDims = dimsIn;

	columnMajor = isColumnMajor;

	image = new HostPixelType[imageDims.product()];
}

ImageContainer::ImageContainer( HostPixelType* imageIn, Vec<unsigned int> dims, bool isColumnMajor/*=false*/)
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

HostPixelType ImageContainer::getPixelValue(unsigned int x, unsigned int y, unsigned int z) const
{
	return getPixelValue(Vec<unsigned int>(x,y,z));
}

HostPixelType ImageContainer::getPixelValue(Vec<unsigned int> coordinate) const
{
	return image[imageDims.linearAddressAt(coordinate)];
}

void ImageContainer::setPixelValue(unsigned int x, unsigned int y, unsigned int z, unsigned char val)
{
	setPixelValue(Vec<unsigned int>(x,y,z),val);
}

void ImageContainer::setPixelValue(Vec<unsigned int> coordinate, unsigned char val)
{
	image[imageDims.linearAddressAt(coordinate)] = val;
}

const HostPixelType* ImageContainer::getConstROIData (unsigned int minX, unsigned int sizeX, unsigned int minY,
													  unsigned int sizeY, unsigned int minZ, unsigned int sizeZ) const
{
	return getConstROIData(Vec<unsigned int>(minX,minY,minZ), Vec<unsigned int>(sizeX,sizeY,sizeZ));
}

const HostPixelType* ImageContainer::getConstROIData(Vec<unsigned int> startIndex, Vec<unsigned int> size) const
{
	HostPixelType* imageOut = new HostPixelType[size.product()];

	unsigned int i=0;
	Vec<unsigned int> curIdx(startIndex);
	for (curIdx.z=startIndex.z; curIdx.z<size.z; ++curIdx.z)
	{
		for (curIdx.y=startIndex.y; curIdx.y<size.y; ++curIdx.y)
		{
			for (curIdx.x=startIndex.x; curIdx.x<size.x; ++curIdx.x)		
			{
				imageOut[i] = getPixelValue(curIdx);
				++i;
			}
		}
	}

	return imageOut;
}

void ImageContainer::loadImage( const HostPixelType* imageIn, Vec<unsigned int> dims, bool isColumnMajor/*=false*/ )
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
		unsigned int i = 0;
		Vec<unsigned int> curInIdx(0,0,0);
		for (curInIdx.z=0; curInIdx.z<dims.z; ++curInIdx.z)
		{
			for (curInIdx.y=0; curInIdx.y<dims.y; ++curInIdx.y)
			{
				for (curInIdx.x=0; curInIdx.x<dims.x; ++curInIdx.x)
				{
					if (i>=imageDims.product())
						break;

					image[i] = imageIn[imageDims.linearAddressAt(curInIdx,true)];
					++i;
				}
			}
		}
		columnMajor = false;
	}
}

void ImageContainer::loadImage( const HostPixelType* imageIn, unsigned int width, unsigned int height, unsigned int depth, bool isColumnMajor/*=false*/ )
{
	loadImage(imageIn,Vec<unsigned int>(width,height,depth),isColumnMajor);
}

HostPixelType& ImageContainer::operator[]( Vec<unsigned int> coordinate )
{
	return image[imageDims.linearAddressAt(coordinate,columnMajor)];
}

const HostPixelType& ImageContainer::operator[]( Vec<unsigned int> coordinate ) const
{
	return image[imageDims.linearAddressAt(coordinate,columnMajor)];
}

HostPixelType& ImageContainer::at( Vec<unsigned int> coordinate )
{
	return image[imageDims.linearAddressAt(coordinate,columnMajor)];
}

void ImageContainer::setROIData( HostPixelType* imageIn, Vec<unsigned int> startIndex, Vec<unsigned int> sizeIn )
{
	Vec<unsigned int> curIdx(startIndex);
	for (curIdx.z=startIndex.z; curIdx.z<startIndex.z+sizeIn.z && curIdx.z<imageDims.z; ++curIdx.z)
	{
		for (curIdx.y=startIndex.y; curIdx.y<startIndex.y+sizeIn.y && curIdx.y<imageDims.y; ++curIdx.y)
		{
			for (curIdx.x=startIndex.x; curIdx.x<startIndex.x+sizeIn.x && curIdx.x<imageDims.x; ++curIdx.x)		
			{
				setPixelValue(curIdx,imageIn[sizeIn.linearAddressAt(curIdx-startIndex)]);
			}
		}
	}
}

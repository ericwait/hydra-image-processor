#include "CudaImageContainer.cuh"

CudaImageContainer::CudaImageContainer(size_t width, size_t height, size_t depth, bool isColumnMajor/*=false*/)
{
	defaults();
	imageDims.x = width;
	imageDims.y = height;
	imageDims.z = depth;

	columnMajor = isColumnMajor;

	image = new DevicePixelType[imageDims.product()];
}

CudaImageContainer::CudaImageContainer(DeviceVec<size_t> dimsIn, bool isColumnMajor/*=false*/)
{
	defaults();
	imageDims = dimsIn;

	columnMajor = isColumnMajor;

	image = new DevicePixelType[imageDims.product()];
}

CudaImageContainer::CudaImageContainer( DevicePixelType* imageIn, DeviceVec<size_t> dims, bool isColumnMajor/*=false*/)
{
	defaults();
	columnMajor = isColumnMajor;
	loadImage(imageIn,dims,isColumnMajor);
}

void CudaImageContainer::copy(const CudaImageContainer& im)
{
	clear();

	image = new DevicePixelType[imageDims.product()];
	memcpy((void*)image,(void*)(im.getConstMemoryPointer()),sizeof(DevicePixelType)*imageDims.product());
}

void CudaImageContainer::clear()
{
	if (image!=NULL)
	{
		delete[] image;
		image = NULL;
	}

	defaults();
}

DevicePixelType CudaImageContainer::getPixelValue(size_t x, size_t y, size_t z) const
{
	return getPixelValue(DeviceVec<size_t>(x,y,z));
}

DevicePixelType CudaImageContainer::getPixelValue(DeviceVec<size_t> coordinate) const
{
	return image[imageDims.linearAddressAt(coordinate)];
}

void CudaImageContainer::setPixelValue(size_t x, size_t y, size_t z, unsigned char val)
{
	setPixelValue(DeviceVec<size_t>(x,y,z),val);
}

void CudaImageContainer::setPixelValue(DeviceVec<size_t> coordinate, unsigned char val)
{
	image[imageDims.linearAddressAt(coordinate)] = val;
}

const DevicePixelType* CudaImageContainer::getConstROIData (size_t minX, size_t sizeX, size_t minY,
													  size_t sizeY, size_t minZ, size_t sizeZ) const
{
	return getConstROIData(DeviceVec<size_t>(minX,minY,minZ), DeviceVec<size_t>(sizeX,sizeY,sizeZ));
}

const DevicePixelType* CudaImageContainer::getConstROIData(DeviceVec<size_t> startIndex, DeviceVec<size_t> size) const
{
	DevicePixelType* imageOut = new DevicePixelType[size.product()];

	size_t i=0;
	DeviceVec<size_t> curIdx(startIndex);
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

void CudaImageContainer::loadImage( const DevicePixelType* imageIn, DeviceVec<size_t> dims, bool isColumnMajor/*=false*/ )
{
	if (!isColumnMajor)
	{
		if (dims!=imageDims)
		{
			if (image!=NULL)
			{
				delete[] image;
			}
			image = new DevicePixelType[dims.product()];
			imageDims = dims;
		}

		memcpy(image,imageIn,sizeof(DevicePixelType)*imageDims.product());
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
			image = new DevicePixelType[dims.product()];
			imageDims = dims;
		}
		//TODO: take this out when the cuda storage can take column major buffers
		size_t i = 0;
		double acum = 0.0;
		int mx = -1;
		DeviceVec<size_t> curInIdx(0,0,0);
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

void CudaImageContainer::loadImage( const DevicePixelType* imageIn, size_t width, size_t height, size_t depth, bool isColumnMajor/*=false*/ )
{
	loadImage(imageIn,DeviceVec<size_t>(width,height,depth),isColumnMajor);
}

DevicePixelType& CudaImageContainer::operator[]( DeviceVec<size_t> coordinate )
{
	return image[imageDims.linearAddressAt(coordinate,columnMajor)];
}

const DevicePixelType& CudaImageContainer::operator[]( DeviceVec<size_t> coordinate ) const
{
	return image[imageDims.linearAddressAt(coordinate,columnMajor)];
}

DevicePixelType& CudaImageContainer::at( DeviceVec<size_t> coordinate )
{
	return image[imageDims.linearAddressAt(coordinate,columnMajor)];
}

void CudaImageContainer::setROIData( DevicePixelType* imageIn, DeviceVec<size_t> startIndex, DeviceVec<size_t> sizeIn )
{
	DeviceVec<size_t> curIdx(0,0,0);
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

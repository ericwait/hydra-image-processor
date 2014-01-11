#ifndef IMAGE_CONTAINER_DESTRUCT_H
#define IMAGE_CONTAINER_DESTRUCT_H

#include "ImageContainer.h"

class ImageContainerDestruct : public ImageContainer
{
public:
	ImageContainerDestruct(size_t width, size_t height, size_t depth, bool isColumnMajor=false) : ImageContainer(width,height,depth){};
	ImageContainerDestruct(Vec<size_t> dims, bool isColumnMajor=false) : ImageContainer(dims,isColumnMajor){};
	ImageContainerDestruct(HostPixelType* imageIn, Vec<size_t> dims, bool isColumnMajor=false) : ImageContainer(imageIn,dims,isColumnMajor){};
	ImageContainerDestruct(const ImageContainerDestruct& image) : ImageContainer(image){};
	virtual ~ImageContainerDestruct();

protected:
	ImageContainerDestruct() : ImageContainer(){};
};
#endif
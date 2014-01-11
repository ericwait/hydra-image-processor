#include "ImageContainerDestruct.h"

ImageContainerDestruct::~ImageContainerDestruct()
{
	if (image!=NULL)
	{
		delete[] image;
		image = NULL;
	}
}
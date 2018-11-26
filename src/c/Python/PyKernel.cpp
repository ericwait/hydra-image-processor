#include "PyKernel.h"

#include "../Cuda/Vec.h"
#include "../Cuda/ImageDimensions.cuh"

#include <string.h>


ImageContainer<float> getKernel(PyArrayObject* kernel)
{
	int numDims = PyArray_NDIM(kernel);
	const npy_intp* DIMS = PyArray_DIMS(kernel);

	Vec<std::size_t> kernDims(1);

	if ( numDims > 2 )
		kernDims.z = (std::size_t)DIMS[2];
	else
		kernDims.z = 1;

	if ( numDims > 1 )
		kernDims.y = (std::size_t)DIMS[1];
	else
		kernDims.y = 1;

	if ( numDims > 0 )
		kernDims.x = (std::size_t)DIMS[0];
	else
		return ImageContainer<float>();

	ImageContainer<float> kernelOut;
	kernelOut.resize(ImageDimensions(kernDims, 1, 1));

	float* kernPtr = kernelOut.getPtr();

	
	if ( PyArray_TYPE(kernel) == NPY_BOOL )
	{
		bool* mexKernelData;
		mexKernelData = (bool*)PyArray_DATA(kernel);

		for ( int i = 0; i < kernDims.product(); ++i )
		{
			if ( mexKernelData[i] )
				kernPtr[i] = 1.0f;
			else
				kernPtr[i] = 0.0f;
		}
	}
	else if ( PyArray_TYPE(kernel) == NPY_UINT8 )
	{
		unsigned char* mexKernelData;
		mexKernelData = (unsigned char*)PyArray_DATA(kernel);

		for ( int i = 0; i < kernDims.product(); ++i )
		{
			double val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}
	else if ( PyArray_TYPE(kernel) == NPY_INT16 )
	{
		short* mexKernelData;
		mexKernelData = (short*)PyArray_DATA(kernel);

		for ( int i = 0; i < kernDims.product(); ++i )
		{
			short val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}
	else if ( PyArray_TYPE(kernel) == NPY_UINT16 )
	{
		unsigned short* mexKernelData;
		mexKernelData = (unsigned short*)PyArray_DATA(kernel);

		for ( int i = 0; i < kernDims.product(); ++i )
		{
			unsigned short val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}
	else if ( PyArray_TYPE(kernel) == NPY_INT32 )
	{
		int* mexKernelData;
		mexKernelData = (int*)PyArray_DATA(kernel);

		for ( int i = 0; i < kernDims.product(); ++i )
		{
			int val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}
	else if ( PyArray_TYPE(kernel) == NPY_UINT32 )
	{
		unsigned int* mexKernelData;
		mexKernelData = (unsigned int*)PyArray_DATA(kernel);

		for ( int i = 0; i < kernDims.product(); ++i )
		{
			unsigned int val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}
	else if ( PyArray_TYPE(kernel) == NPY_FLOAT )
	{
		float* mexKernelData = (float*)PyArray_DATA(kernel);
		memcpy(kernPtr, mexKernelData, sizeof(float)*kernDims.product());
	}
	else if ( PyArray_TYPE(kernel) == NPY_DOUBLE )
	{
		double* mexKernelData;
		mexKernelData = (double*)PyArray_DATA(kernel);

		for ( int i = 0; i < kernDims.product(); ++i )
		{
			double val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}

	return kernelOut;
}

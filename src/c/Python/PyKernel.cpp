#include "PyKernel.h"

#include "../Cuda/Vec.h"
#include "../Cuda/ImageDimensions.cuh"

#include <string.h>

template <typename T>
void convertKernelImage(ImageView<float> kernel, PyArrayObject* inKernel)
{
	float* outPtr = kernel.getPtr();
	T* kernData = (T*) PyArray_DATA(inKernel);

	for ( int i=0; i < kernel.getNumElements(); ++i )
		outPtr[i] = static_cast<float>(kernData[i]);
}

template <>
void convertKernelImage<float>(ImageView<float> kernel, PyArrayObject* inKernel)
{
	float* outPtr = kernel.getPtr();
	float* kernData = (float*)PyArray_DATA(inKernel);

	memcpy(outPtr, kernData, kernel.getNumElements()*sizeof(float));
}

template <>
void convertKernelImage<bool>(ImageView<float> kernel, PyArrayObject* inKernel)
{
	float* outPtr = kernel.getPtr();
	bool* kernData = (bool*)PyArray_DATA(inKernel);

	for ( int i=0; i < kernel.getNumElements(); ++i )
		outPtr[i] = (kernData[i]) ? (1.0f) : (0.0f);
}


ImageOwner<float> getKernel(PyArrayObject* kernel)
{
	Script::DimInfo info = Script::getDimInfo(kernel);

	if ( info.dims.size() < 1 )
		return ImageOwner<float>();

	Vec<std::size_t> kernDims(1);
	std::size_t chkDims = std::min<int>((int)info.dims.size(),3);
	for ( int i=0; i < chkDims; ++i )
		kernDims.e[i] = info.dims[i];

	ImageOwner<float> kernelOut(kernDims, 1, 1);
	if ( PyArray_TYPE(kernel) == NPY_BOOL )
	{
		convertKernelImage<bool>(kernelOut, kernel);
	}
	else if ( PyArray_TYPE(kernel) == NPY_UINT8 )
	{
		convertKernelImage<uint8_t>(kernelOut, kernel);
	}
	else if ( PyArray_TYPE(kernel) == NPY_UINT16 )
	{
		convertKernelImage<uint16_t>(kernelOut, kernel);
	}
	else if ( PyArray_TYPE(kernel) == NPY_INT16 )
	{
		convertKernelImage<int16_t>(kernelOut, kernel);
	}
	else if ( PyArray_TYPE(kernel) == NPY_UINT32 )
	{
		convertKernelImage<uint32_t>(kernelOut, kernel);
	}
	else if ( PyArray_TYPE(kernel) == NPY_INT32 )
	{
		convertKernelImage<int32_t>(kernelOut, kernel);
	}
	else if ( PyArray_TYPE(kernel) == NPY_FLOAT )
	{
		convertKernelImage<float>(kernelOut, kernel);
	}
	else if ( PyArray_TYPE(kernel) == NPY_DOUBLE )
	{
		convertKernelImage<double>(kernelOut, kernel);
	}

	return kernelOut;
}

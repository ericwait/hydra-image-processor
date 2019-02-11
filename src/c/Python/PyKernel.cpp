#include "PyKernel.h"

#include "../Cuda/Vec.h"
#include "../Cuda/ImageDimensions.cuh"

#include <string.h>

template <typename T>
void convertKernelImage(ImageView<float> kernel, Script::ArrayType* inKernel)
{
	float* outPtr = kernel.getPtr();
	T* kernData = Script::ArrayInfo::getData<T>(inKernel);

	for ( int i=0; i < kernel.getNumElements(); ++i )
		outPtr[i] = static_cast<float>(kernData[i]);
}

template <>
void convertKernelImage<float>(ImageView<float> kernel, Script::ArrayType* inKernel)
{
	float* outPtr = kernel.getPtr();
	float* kernData = Script::ArrayInfo::getData<float>(inKernel);

	memcpy(outPtr, kernData, kernel.getNumElements()*sizeof(float));
}

template <>
void convertKernelImage<bool>(ImageView<float> kernel, Script::ArrayType* inKernel)
{
	float* outPtr = kernel.getPtr();
	bool* kernData = Script::ArrayInfo::getData<bool>(inKernel);

	for ( int i=0; i < kernel.getNumElements(); ++i )
		outPtr[i] = (kernData[i]) ? (1.0f) : (0.0f);
}


ImageOwner<float> getKernel(PyArrayObject* kernel)
{
	Script::DimInfo info = Script::getDimInfo(kernel);

	if ( info.dims.size() < 1 )
		return ImageOwner<float>();

	ImageDimensions kernDims = Script::makeImageDims(info);
	kernDims.chan = 1;
	kernDims.frame = 1;

	ImageOwner<float> kernelOut(kernDims);
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

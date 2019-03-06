#include "MexKernel.h"

#include "../Cuda/Vec.h"
#include "../Cuda/ImageDimensions.cuh"

#include <string.h>


template <typename T>
void convertKernelImage(ImageView<float> kernel, const Script::ArrayType* inKernel)
{
	float* outPtr = kernel.getPtr();
	T* kernData = Script::Array::getData<T>(inKernel);

	for ( int i=0; i < kernel.getNumElements(); ++i )
		outPtr[i] = static_cast<float>(kernData[i]);
}

template <>
void convertKernelImage<float>(ImageView<float> kernel, const Script::ArrayType* inKernel)
{
	float* outPtr = kernel.getPtr();
	float* kernData = Script::Array::getData<float>(inKernel);

	std::memcpy(outPtr, kernData, kernel.getNumElements()*sizeof(float));
}

template <>
void convertKernelImage<bool>(ImageView<float> kernel, const Script::ArrayType* inKernel)
{
	float* outPtr = kernel.getPtr();
	bool* kernData = Script::Array::getData<bool>(inKernel);

	for ( int i=0; i < kernel.getNumElements(); ++i )
		outPtr[i] = (kernData[i]) ? (1.0f) : (0.0f);
}


ImageOwner<float> getKernel(const mxArray* mexKernel)
{
	Script::DimInfo info = Script::getDimInfo(mexKernel);

	if ( info.dims.size() < 1 || info.dims.size() > 3 )
		return ImageOwner<float>();

	ImageDimensions kernDims = Script::makeImageDims(info);
	kernDims.chan = 1;
	kernDims.frame = 1;

	ImageOwner<float> kernelOut(kernDims);
	if (mxIsLogical(mexKernel))
	{
		convertKernelImage<bool>(kernelOut, mexKernel);
	}
	else if (mxIsUint8(mexKernel))
	{
		convertKernelImage<bool>(kernelOut, mexKernel);
	}
	else if (mxIsInt16(mexKernel))
	{
		convertKernelImage<bool>(kernelOut, mexKernel);
	}
	else if (mxIsUint16(mexKernel))
	{
		convertKernelImage<bool>(kernelOut, mexKernel);
	}
	else if (mxIsInt32(mexKernel))
	{
		convertKernelImage<bool>(kernelOut, mexKernel);
	}
	else if (mxIsUint32(mexKernel))
	{
		convertKernelImage<bool>(kernelOut, mexKernel);
	}
	else if (mxIsSingle(mexKernel))
	{
		convertKernelImage<float>(kernelOut, mexKernel);
	}
	else if (mxIsDouble(mexKernel))
	{
		convertKernelImage<double>(kernelOut, mexKernel);
	}

	return kernelOut;
}


#include "MexKernel.h"
#include "../Cuda/Vec.h"
#include "../Cuda/ImageDimensions.cuh"

#include <string.h>

ImageOwner<float> getKernel(const mxArray* mexKernel)
{
	std::size_t numDims = mxGetNumberOfDimensions(mexKernel);
	const mwSize* DIMS = mxGetDimensions(mexKernel);

	Vec<std::size_t> kernDims(1);

	if (numDims > 2)
		kernDims.z = (std::size_t)DIMS[2];
	else
		kernDims.z = 1;

	if (numDims > 1)
		kernDims.y = (std::size_t)DIMS[1];
	else
		kernDims.y = 1;

	if (numDims > 0)
		kernDims.x = (std::size_t)DIMS[0];
	else
		return ImageOwner<float>();

	ImageOwner<float> kernelOut(kernDims, 1, 1);
	float* kernPtr = kernelOut.getPtr();

	if (mxIsLogical(mexKernel))
	{
		bool* mexKernelData;
		mexKernelData = (bool*)mxGetData(mexKernel);

		for (int i = 0; i < kernDims.product(); ++i)
		{
			if (mexKernelData[i])
				kernPtr[i] = 1.0f;
			else
				kernPtr[i] = 0.0f;
		}
	}
	else if (mxIsUint8(mexKernel))
	{
		unsigned char* mexKernelData;
		mexKernelData = (unsigned char*)mxGetData(mexKernel);

		for (int i = 0; i < kernDims.product(); ++i)
		{
			double val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}
	else if (mxIsInt16(mexKernel))
	{
		short* mexKernelData;
		mexKernelData = (short*)mxGetData(mexKernel);

		for (int i = 0; i < kernDims.product(); ++i)
		{
			short val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}
	else if (mxIsUint16(mexKernel))
	{
		unsigned short* mexKernelData;
		mexKernelData = (unsigned short*)mxGetData(mexKernel);

		for (int i = 0; i < kernDims.product(); ++i)
		{
			unsigned short val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}
	else if (mxIsInt32(mexKernel))
	{
		int* mexKernelData;
		mexKernelData = (int*)mxGetData(mexKernel);

		for (int i = 0; i < kernDims.product(); ++i)
		{
			int val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}
	else if (mxIsUint32(mexKernel))
	{
		unsigned int* mexKernelData;
		mexKernelData = (unsigned int*)mxGetData(mexKernel);

		for (int i = 0; i < kernDims.product(); ++i)
		{
			unsigned int val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}
	else if (mxIsSingle(mexKernel))
	{
		float* mexKernelData = (float*)mxGetData(mexKernel);
		memcpy(kernPtr, mexKernelData, sizeof(float)*kernDims.product());
	}
	else if (mxIsDouble(mexKernel))
	{
		double* mexKernelData;
		mexKernelData = (double*)mxGetData(mexKernel);

		for (int i = 0; i < kernDims.product(); ++i)
		{
			double val = mexKernelData[i];
			kernPtr[i] = (float)val;
		}
	}

	return kernelOut;
}


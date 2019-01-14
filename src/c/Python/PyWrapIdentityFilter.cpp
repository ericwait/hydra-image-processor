#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

#include "PyKernel.h"



const char PyWrapIdentityFilter::docString[] = "imageOut = HIP.IdentityFilter(imageIn,[device])\n\n"\
"Identity Filter for testing. Copies image data to GPU memory and back into output image.\n"\
"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"\
"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"\
"\t\thow to stride or jump to the next spatial block.\n"\
"\n"\
"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"\
"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"\
"\t\tthe data across multiple devices.\n"\
"\n"\
"\timageOut = This will be an array of the same type and shape as the input array.\n";

template <typename T>
void PyWrapIdentity_run(const PyArrayObject* inIm, PyArrayObject** outIm, int device)
{
	Script::DimInfo inInfo = Script::getDimInfo(inIm);
	ImageView<T> imageIn = Script::wrapInputImage<T>(inIm, inInfo);
	ImageView<T> imageOut = Script::createOutputImage<T>(outIm, inInfo);

	identityFilter(imageIn, imageOut, device);
}


PyObject* PyWrapIdentityFilter::execute(PyObject* self, PyObject* args)
{
	int device = -1;

	PyArrayObject* imIn;

	if ( !PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &imIn, &device) )
		return nullptr;

	// TODO: These checks should be superfluous
	if ( imIn == nullptr ) return nullptr;

	if ( !Script::ArrayInfo::isContiguous(imIn) )
	{
		PyErr_SetString(PyExc_RuntimeError, "Input image must be a contiguous numpy array!");
		return nullptr;
	}

	PyArrayObject* imOut = nullptr;
	if ( PyArray_TYPE(imIn) == NPY_BOOL )
	{
		PyWrapIdentity_run<bool>(imIn, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT8 )
	{
		PyWrapIdentity_run<uint8_t>(imIn, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT16 )
	{
		PyWrapIdentity_run<uint16_t>(imIn, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT16 )
	{
		PyWrapIdentity_run<int16_t>(imIn, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT32 )
	{
		PyWrapIdentity_run<uint32_t>(imIn, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT32 )
	{
		PyWrapIdentity_run<int32_t>(imIn, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_FLOAT )
	{
		PyWrapIdentity_run<float>(imIn, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_DOUBLE )
	{
		PyWrapIdentity_run<double>(imIn, &imOut, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");
		return nullptr;
	}

	return ((PyObject*)imOut);
}

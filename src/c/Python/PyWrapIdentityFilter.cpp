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
	T* imageInPtr;
	T* imageOutPtr;

	ImageDimensions imageDims;
	Script::setupImagePointers(inIm, &imageInPtr, imageDims, outIm, &imageOutPtr);

	ImageView<T> imageIn(imageInPtr, imageDims);
	ImageView<T> imageOut(imageOutPtr, imageDims);

	identityFilter(imageIn, imageOut, device);
}


PyObject* PyWrapIdentityFilter::execute(PyObject* self, PyObject* args)
{
	int device = -1;

	PyObject* imIn;

	if ( !PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &imIn, &device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;

	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* imOut = nullptr;

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		PyWrapIdentity_run<bool>(imContig, &imOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		PyWrapIdentity_run<uint8_t>(imContig, &imOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		PyWrapIdentity_run<uint16_t>(imContig, &imOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		PyWrapIdentity_run<int16_t>(imContig, &imOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		PyWrapIdentity_run<uint32_t>(imContig, &imOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		PyWrapIdentity_run<int32_t>(imContig, &imOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		PyWrapIdentity_run<float>(imContig, &imOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		PyWrapIdentity_run<double>(imContig, &imOut, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");

		Py_XDECREF(imContig);

		return nullptr;
	}

	Py_XDECREF(imContig);

	return ((PyObject*)imOut);
}

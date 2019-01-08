#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"

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


PyObject* PyWrapIdentityFilter::execute(PyObject* self, PyObject* args)
{
	int device = -1;

	PyObject* imIn;

	if ( !PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &imIn, &device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;

	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

	ImageDimensions imageDims;
	PyArrayObject* imOut = nullptr;

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		bool* imageInPtr, *imageOutPtr;

		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<bool> imageIn(imageInPtr, imageDims);
		ImageContainer<bool> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);

	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		unsigned char* imageInPtr, *imageOutPtr;

		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<unsigned char> imageIn(imageInPtr, imageDims);
		ImageContainer<unsigned char> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		unsigned short* imageInPtr, *imageOutPtr;

		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<unsigned short> imageIn(imageInPtr, imageDims);
		ImageContainer<unsigned short> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		short* imageInPtr, *imageOutPtr;

		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<short> imageIn(imageInPtr, imageDims);
		ImageContainer<short> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		unsigned int* imageInPtr, *imageOutPtr;

		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<unsigned int> imageIn(imageInPtr, imageDims);
		ImageContainer<unsigned int> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		int* imageInPtr, *imageOutPtr;

		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<int> imageIn(imageInPtr, imageDims);
		ImageContainer<int> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		float* imageInPtr, *imageOutPtr;

		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<float> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		double* imageInPtr, *imageOutPtr;

		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<double> imageIn(imageInPtr, imageDims);
		ImageContainer<double> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");

		Py_XDECREF(imContig);
		Py_XDECREF(imOut);

		return nullptr;
	}

	Py_XDECREF(imContig);

	return ((PyObject*)imOut);
}

#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"

#include "PyKernel.h"


const char PyWrapLoG::docString[] = "imageOut = HIP.LoG(imageIn,Sigmas,[device])\n\n"\
"Apply a Lapplacian of Gaussian filter with the given sigmas.\n"\
"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"\
"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"\
"\t\thow to stride or jump to the next spatial block.\n"\
"\n"\
"\tSigmas = This should be an array of three positive values that represent the standard deviation of a Gaussian curve.\n"\
"\t\tZeros (0) in this array will not smooth in that direction.\n"\
"\n"\
"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"\
"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"\
"\t\tthe data across multiple devices.\n"\
"\n"\
"\timageOut = This will be an array of the same type and shape as the input array.\n";


PyObject* PyWrapLoG::execute(PyObject* self, PyObject* args)
{
	int device = -1;

	PyObject* imIn;
	PyObject* inSigmas;

	if ( !PyArg_ParseTuple(args, "O!O|i", &PyArray_Type, &imIn, &inSigmas, &device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;


	Vec<double> sigmas;
	if ( !pyobjToVec(inSigmas, sigmas) )
	{
		PyErr_SetString(PyExc_TypeError, "Sigmas must be a 3-element numeric list");
		return nullptr;
	}

	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

	ImageDimensions imageDims;
	PyArrayObject* imOut = nullptr;

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		bool* imageInPtr;
		float* imageOutPtr;

		setupInputPointers(imContig, imageDims, &imageInPtr);
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<bool> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);

	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		unsigned char* imageInPtr;
		float* imageOutPtr;

		setupInputPointers(imContig, imageDims, &imageInPtr);
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<unsigned char> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		unsigned short* imageInPtr;
		float* imageOutPtr;

		setupInputPointers(imContig, imageDims, &imageInPtr);
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<unsigned short> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		short* imageInPtr;
		float* imageOutPtr;

		setupInputPointers(imContig, imageDims, &imageInPtr);
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<short> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		unsigned int* imageInPtr;
		float* imageOutPtr;

		setupInputPointers(imContig, imageDims, &imageInPtr);
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<unsigned int> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		int* imageInPtr;
		float* imageOutPtr;

		setupInputPointers(imContig, imageDims, &imageInPtr);
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<int> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		float* imageInPtr;
		float* imageOutPtr;

		setupInputPointers(imContig, imageDims, &imageInPtr);
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<float> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		double* imageInPtr;
		float* imageOutPtr;

		setupInputPointers(imContig, imageDims, &imageInPtr);
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		// TODO: Do we really want to use float outputs here?
		ImageContainer<double> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
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

#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"

#include "PyKernel.h"


const char PyWrapWienerFilter::docString[] = "imageOut = HIP.WienerFilter(imageIn,kernel,[tnoiseVariance],[device])\n\n"\
"A Wiener filter aims to denoise an image in a linear fashion.\n"\
"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"\
"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"\
"\t\thow to stride or jump to the next spatial block.\n"\
"\n"\
"\tkernel (optional) = This is a one to three dimensional array that will be used to determine neighborhood operations.\n"\
"\t\tIn this case, the positions in the kernel that do not equal zeros will be evaluated.\n"\
"\t\tIn other words, this can be viewed as a structuring element for the neighborhood.\n"\
"\t\t This can be an empty array [] and which will use a 3x3x3 neighborhood (or equivalent given input dimension).\n"\
"\n"\
"\tnoiseVariance (optional) =  This is the expected variance of the noise.\n"\
"\t\tThis should be a scalar value or an empty array [].\n"\
"\n"\
"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"\
"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"\
"\t\tthe data across multiple devices.\n"\
"\n"\
"\timageOut = This will be an array of the same type and shape as the input array.\n";


PyObject* PyWrapWienerFilter::execute(PyObject* self, PyObject* args)
{
	PyObject* imIn;
	PyObject* inKern;

	double noiseVar = -1.0;
	int device = -1;

	if ( !PyArg_ParseTuple(args, "O!O!|di", &PyArray_Type, &imIn, &PyArray_Type, &inKern,
							&noiseVar, &device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;
	if ( inKern == nullptr ) return nullptr;

	PyArrayObject* kernContig = (PyArrayObject*)PyArray_FROM_OTF(inKern, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

	ImageContainer<float> kernel = getKernel(kernContig);
	Py_XDECREF(kernContig);

	if ( kernel.getDims().getNumElements() == 0 )
	{
		kernel.clear();

		PyErr_SetString(PyExc_RuntimeError, "Unable to create kernel");
		return nullptr;
	}

	ImageDimensions imageDims;
	PyArrayObject* imOut = nullptr;

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		bool* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<bool> imageIn(imageInPtr, imageDims);
		ImageContainer<bool> imageOut(imageOutPtr, imageDims);

		wienerFilter(imageIn, imageOut, kernel, noiseVar, device);

	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		unsigned char* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<unsigned char> imageIn(imageInPtr, imageDims);
		ImageContainer<unsigned char> imageOut(imageOutPtr, imageDims);

		wienerFilter(imageIn, imageOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		unsigned short* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<unsigned short> imageIn(imageInPtr, imageDims);
		ImageContainer<unsigned short> imageOut(imageOutPtr, imageDims);

		wienerFilter(imageIn, imageOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		short* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<short> imageIn(imageInPtr, imageDims);
		ImageContainer<short> imageOut(imageOutPtr, imageDims);

		wienerFilter(imageIn, imageOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		unsigned int* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<unsigned int> imageIn(imageInPtr, imageDims);
		ImageContainer<unsigned int> imageOut(imageOutPtr, imageDims);

		wienerFilter(imageIn, imageOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		int* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<int> imageIn(imageInPtr, imageDims);
		ImageContainer<int> imageOut(imageOutPtr, imageDims);

		wienerFilter(imageIn, imageOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		float* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<float> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		wienerFilter(imageIn, imageOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		double* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims, &imOut, &imageOutPtr);

		ImageContainer<double> imageIn(imageInPtr, imageDims);
		ImageContainer<double> imageOut(imageOutPtr, imageDims);

		wienerFilter(imageIn, imageOut, kernel, noiseVar, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");

		Py_XDECREF(imContig);
		Py_XDECREF(imOut);

		return nullptr;
	}

	Py_XDECREF(imContig);

	kernel.clear();

	return ((PyObject*)imOut);
}

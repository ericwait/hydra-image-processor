#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"

#include "PyKernel.h"

// TODO: This is a superfluous function!!!!

const char PyWrapMinMax::docString[] = "minOut,maxOut = HIP.MinMax(imageIn,[device])\n\n"\
"This returns the global min and max values.\n"\
"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"\
"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"\
"\t\thow to stride or jump to the next spatial block.\n"\
"\n"\
"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"\
"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"\
"\t\tthe data across multiple devices.\n"\
"\n"\
"\tminOut = This is the minimum value found in the input.\n"\
"\tmaxOut = This is the maximum value found in the input.\n";

PyObject* PyWrapMinMax::execute(PyObject* self, PyObject* args)
{
	PyObject* imIn;
	PyObject* inKern;

	int device = -1;

	if ( !PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &imIn, &device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;

	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

	ImageDimensions imageDims;
	PyObject* outMinMax = PyTuple_New(2);

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		bool* imageInPtr;
		bool minVal, maxVal;

		setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<bool> imageIn(imageInPtr, imageDims);

		minMax(imageIn, minVal, maxVal, device);

		PyTuple_SetItem(outMinMax, 0, PyBool_FromLong(minVal));
		PyTuple_SetItem(outMinMax, 1, PyBool_FromLong(maxVal));
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		unsigned char* imageInPtr;
		unsigned char minVal, maxVal;

		setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<unsigned char> imageIn(imageInPtr, imageDims);

		minMax(imageIn, minVal, maxVal, device);

		PyTuple_SetItem(outMinMax, 0, PyLong_FromLong(minVal));
		PyTuple_SetItem(outMinMax, 1, PyLong_FromLong(maxVal));
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		unsigned short* imageInPtr;
		unsigned short minVal, maxVal;

		setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<unsigned short> imageIn(imageInPtr, imageDims);

		minMax(imageIn, minVal, maxVal, device);

		PyTuple_SetItem(outMinMax, 0, PyLong_FromLong(minVal));
		PyTuple_SetItem(outMinMax, 1, PyLong_FromLong(maxVal));
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		short* imageInPtr, minVal, maxVal;

		setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<short> imageIn(imageInPtr, imageDims);

		minMax(imageIn, minVal, maxVal, device);

		PyTuple_SetItem(outMinMax, 0, PyLong_FromLong(minVal));
		PyTuple_SetItem(outMinMax, 1, PyLong_FromLong(maxVal));
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		unsigned int* imageInPtr;
		unsigned int minVal, maxVal;

		setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<unsigned int> imageIn(imageInPtr, imageDims);

		minMax(imageIn, minVal, maxVal, device);

		PyTuple_SetItem(outMinMax, 0, PyLong_FromLong(minVal));
		PyTuple_SetItem(outMinMax, 1, PyLong_FromLong(maxVal));
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		int* imageInPtr;
		int minVal, maxVal;

		setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<int> imageIn(imageInPtr, imageDims);

		minMax(imageIn, minVal, maxVal, device);

		PyTuple_SetItem(outMinMax, 0, PyLong_FromLong(minVal));
		PyTuple_SetItem(outMinMax, 1, PyLong_FromLong(maxVal));
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		float* imageInPtr;
		float minVal, maxVal;

		setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<float> imageIn(imageInPtr, imageDims);

		minMax(imageIn, minVal, maxVal, device);

		PyTuple_SetItem(outMinMax, 0, PyFloat_FromDouble(minVal));
		PyTuple_SetItem(outMinMax, 1, PyFloat_FromDouble(maxVal));
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		double* imageInPtr;
		double minVal, maxVal;

		setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<double> imageIn(imageInPtr, imageDims);

		minMax(imageIn, minVal, maxVal, device);

		PyTuple_SetItem(outMinMax, 0, PyFloat_FromDouble(minVal));
		PyTuple_SetItem(outMinMax, 1, PyFloat_FromDouble(maxVal));
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");

		Py_XDECREF(imContig);
		Py_XDECREF(outMinMax);

		return nullptr;
	}

	Py_XDECREF(imContig);

	return outMinMax;
}

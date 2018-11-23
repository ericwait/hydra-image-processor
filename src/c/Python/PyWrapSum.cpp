#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"

#include "PyKernel.h"


const char PyWrapSum::docString[] = "valueOut = HIP.Sum(imageIn, [device])\n\n"\
"This sums up the entire array in.\n"\
"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"\
"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"\
"\t\thow to stride or jump to the next spatial block.\n"\
"\n"\
"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"\
"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"\
"\t\tthe data across multiple devices.\n"\
"\n"\
"\tvalueOut = This is the summation of the entire array.\n";


PyObject* PyWrapSum::execute(PyObject* self, PyObject* args)
{
	PyObject* imIn;

	int device = -1;

	if ( !PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &imIn, &device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;

	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

	ImageDimensions imageDims;
	PyObject* outSum = nullptr;

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		bool* imageInPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<bool> imageIn(imageInPtr, imageDims);

		std::size_t outVal = 0;
		sum(imageIn, outVal, device);

		outSum = PyLong_FromLongLong(outVal);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		unsigned char* imageInPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<unsigned char> imageIn(imageInPtr, imageDims);

		std::size_t outVal = 0;
		sum(imageIn, outVal, device);

		outSum = PyLong_FromLongLong(outVal);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		unsigned short* imageInPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<unsigned short> imageIn(imageInPtr, imageDims);

		std::size_t outVal = 0;
		sum(imageIn, outVal, device);

		outSum = PyLong_FromLongLong(outVal);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		short* imageInPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<short> imageIn(imageInPtr, imageDims);

		long long outVal = 0;
		sum(imageIn, outVal, device);

		outSum = PyLong_FromLongLong(outVal);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		unsigned int* imageInPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<unsigned int> imageIn(imageInPtr, imageDims);

		std::size_t outVal = 0;
		sum(imageIn, outVal, device);

		outSum = PyLong_FromLongLong(outVal);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		int* imageInPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<int> imageIn(imageInPtr, imageDims);

		long long outVal = 0;
		sum(imageIn, outVal, device);

		outSum = PyLong_FromLongLong(outVal);
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		float* imageInPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<float> imageIn(imageInPtr, imageDims);

		double outVal = 0;
		sum(imageIn, outVal, device);

		outSum = PyFloat_FromDouble(outVal);
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		double* imageInPtr;
		Script::setupImagePointers(imContig, &imageInPtr, imageDims);

		ImageContainer<double> imageIn(imageInPtr, imageDims);

		double outVal = 0;
		sum(imageIn, outVal, device);

		outSum = PyFloat_FromDouble(outVal);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");

		Py_XDECREF(imContig);

		return nullptr;
	}

	Py_XDECREF(imContig);

	return outSum;
}

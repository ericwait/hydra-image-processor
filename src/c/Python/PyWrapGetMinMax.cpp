#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"

#include "PyKernel.h"


const char PyWrapGetMinMax::docString[] = "minValue,maxValue = HIP.GetMinMax(imageIn, [device])\n\n"\
"This function finds the lowest and highest value in the array that is passed in.\n"\
"\timageIn = This is a one to five dimensional array.\n"\
"\n"\
"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"\
"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"\
"\t\tthe data across multiple devices.\n"\
"\n"\
"\tminValue = This is the lowest value found in the array.\n"\
"\tmaxValue = This is the highest value found in the array.\n";

template <typename T>
void PyWrapMinMax_run(const PyArrayObject* inIm, PyObject* outTuple, int device)
{
	T* imageInPtr;
	T minVal;
	T maxVal;

	ImageDimensions imageDims;
	Script::setupImagePointers(inIm, &imageInPtr, imageDims);

	getMinMax(imageInPtr, imageDims.getNumElements(), minVal, maxVal, device);

	PyTuple_SetItem(outTuple, 0, Script::fromNumeric<T>(minVal));
	PyTuple_SetItem(outTuple, 1, Script::fromNumeric<T>(maxVal));
}


PyObject* PyWrapGetMinMax::execute(PyObject* self, PyObject* args)
{
	PyObject* imIn;

	int device = -1;

	if ( !PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &imIn, &device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;

	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

	ImageDimensions imageDims;
	PyObject* outMinMax = PyTuple_New(2);

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		PyWrapMinMax_run<bool>(imContig, outMinMax, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		PyWrapMinMax_run<uint8_t>(imContig, outMinMax, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		PyWrapMinMax_run<uint16_t>(imContig, outMinMax, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		PyWrapMinMax_run<int16_t>(imContig, outMinMax, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		PyWrapMinMax_run<uint32_t>(imContig, outMinMax, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		PyWrapMinMax_run<int32_t>(imContig, outMinMax, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		PyWrapMinMax_run<float>(imContig, outMinMax, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		PyWrapMinMax_run<double>(imContig, outMinMax, device);
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

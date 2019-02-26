#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

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
	T minVal;
	T maxVal;

	Script::DimInfo inInfo = Script::getDimInfo(inIm);
	ImageView<T> imageIn = Script::wrapInputImage<T>(inIm, inInfo);

	getMinMax(imageIn.getConstPtr(), imageIn.getNumElements(), minVal, maxVal, device);

	PyTuple_SetItem(outTuple, 0, Script::Converter::fromNumeric(minVal));
	PyTuple_SetItem(outTuple, 1, Script::Converter::fromNumeric(maxVal));
}


PyObject* PyWrapGetMinMax::execute(PyObject* self, PyObject* args)
{
	PyArrayObject* imIn;

	int device = -1;

	if ( !PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &imIn, &device) )
		return nullptr;

	// TODO: These checks should be superfluous
	if ( imIn == nullptr ) return nullptr;

	if ( !Script::ArrayInfo::isContiguous(imIn) )
	{
		PyErr_SetString(PyExc_RuntimeError, "Input image must be a contiguous numpy array!");
		return nullptr;
	}

	PyObject* outMinMax = PyTuple_New(2);
	if ( PyArray_TYPE(imIn) == NPY_BOOL )
	{
		PyWrapMinMax_run<bool>(imIn, outMinMax, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT8 )
	{
		PyWrapMinMax_run<uint8_t>(imIn, outMinMax, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT16 )
	{
		PyWrapMinMax_run<uint16_t>(imIn, outMinMax, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT16 )
	{
		PyWrapMinMax_run<int16_t>(imIn, outMinMax, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT32 )
	{
		PyWrapMinMax_run<uint32_t>(imIn, outMinMax, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT32 )
	{
		PyWrapMinMax_run<int32_t>(imIn, outMinMax, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_FLOAT )
	{
		PyWrapMinMax_run<float>(imIn, outMinMax, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_DOUBLE )
	{
		PyWrapMinMax_run<double>(imIn, outMinMax, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");

		Py_XDECREF(outMinMax);

		return nullptr;
	}

	return outMinMax;
}

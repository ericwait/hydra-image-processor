#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

#include "PyKernel.h"


const char PyWrapHighPassFilter::docString[] = "imageOut = HIP.Gaussian(imageIn,Sigmas,[device])\n\n"\
"Filters out low frequency by subtracting a Gaussian blurred version of the input based on the sigmas provided.\n"\
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

template <typename T>
void PyWrapHighPass_run(const PyArrayObject* inIm, PyArrayObject** outIm, Vec<double> sigmas, int device)
{
	Script::DimInfo inInfo = Script::getDimInfo(inIm);
	ImageView<T> imageIn = Script::wrapInputImage<T>(inIm, inInfo);
	ImageView<T> imageOut = Script::createOutputImage<T>(outIm, inInfo);

	highPassFilter(imageIn, imageOut, sigmas, device);
}


PyObject* PyWrapHighPassFilter::execute(PyObject* self, PyObject* args)
{
	int device = -1;

	PyArrayObject* imIn;
	PyObject* inSigmas;

	if ( !PyArg_ParseTuple(args, "O!O|i", &PyArray_Type, &imIn, &inSigmas, &device) )
		return nullptr;

	// TODO: These checks should be superfluous
	if ( imIn == nullptr ) return nullptr;

	if ( !Script::ArrayInfo::isContiguous(imIn) )
	{
		PyErr_SetString(PyExc_RuntimeError, "Input image must be a contiguous numpy array!");
		return nullptr;
	}

	Vec<double> sigmas;
	if ( !Script::pyobjToVec(inSigmas, sigmas) )
	{
		PyErr_SetString(PyExc_TypeError, "Sigmas must be a 3-element numeric list");
		return nullptr;
	}

	PyArrayObject* imOut = nullptr;
	if ( PyArray_TYPE(imIn) == NPY_BOOL )
	{
		PyWrapHighPass_run<bool>(imIn, &imOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT8 )
	{
		PyWrapHighPass_run<uint8_t>(imIn, &imOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT16 )
	{
		PyWrapHighPass_run<uint16_t>(imIn, &imOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT16 )
	{
		PyWrapHighPass_run<int16_t>(imIn, &imOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT32 )
	{
		PyWrapHighPass_run<uint32_t>(imIn, &imOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT32 )
	{
		PyWrapHighPass_run<int32_t>(imIn, &imOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_FLOAT )
	{
		PyWrapHighPass_run<float>(imIn, &imOut, sigmas, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_DOUBLE )
	{
		PyWrapHighPass_run<double>(imIn, &imOut, sigmas, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");
		return nullptr;
	}

	return ((PyObject*)imOut);
}

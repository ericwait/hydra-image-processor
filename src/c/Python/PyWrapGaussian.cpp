#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

#include "PyKernel.h"



const char PyWrapGaussian::docString[] = "imageOut = HIP.Gaussian(imageIn,Sigmas,[numIterations],[device])\n\n"\
"Gaussian smoothing.\n"\
"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"\
"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"\
"\t\thow to stride or jump to the next spatial block.\n"\
"\n"\
"\tSigmas = This should be an array of three positive values that represent the standard deviation of a Gaussian curve.\n"\
"\t\tZeros (0) in this array will not smooth in that direction.\n"\
"\n"\
"\tnumIterations (optional) =  This is the number of iterations to run the max filter for a given position.\n"\
"\t\tThis is useful for growing regions by the shape of the structuring element or for very large neighborhoods.\n"\
"\t\tCan be empty an array [].\n"\
"\n"\
"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"\
"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"\
"\t\tthe data across multiple devices.\n"\
"\n"\
"\timageOut = This will be an array of the same type and shape as the input array.\n";

template <typename T>
void PyWrapGaussian_run(const PyArrayObject* inIm, PyArrayObject** outIm, Vec<double> sigmas, int numIterations, int device)
{
	T* imageInPtr;
	T* imageOutPtr;

	ImageDimensions imageDims;
	Script::setupImagePointers(inIm, &imageInPtr, imageDims, outIm, &imageOutPtr);

	ImageView<T> imageIn(imageInPtr, imageDims);
	ImageView<T> imageOut(imageOutPtr, imageDims);

	gaussian(imageIn, imageOut, sigmas, numIterations, device);
}


PyObject* PyWrapGaussian::execute(PyObject* self, PyObject* args)
{
	int device = -1;
	int numIterations = 1;

	PyObject* imIn;
	PyObject* inSigmas;

	if ( !PyArg_ParseTuple(args, "O!O|ii", &PyArray_Type, &imIn, &inSigmas, &numIterations, &device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;


	Vec<double> sigmas;
	if ( !Script::pyobjToVec(inSigmas, sigmas) )
	{
		PyErr_SetString(PyExc_TypeError, "Sigmas must be a 3-element numeric list");
		return nullptr;
	}

	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* imOut = nullptr;

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		PyWrapGaussian_run<bool>(imContig, &imOut, sigmas, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		PyWrapGaussian_run<uint8_t>(imContig, &imOut, sigmas, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		PyWrapGaussian_run<uint16_t>(imContig, &imOut, sigmas, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		PyWrapGaussian_run<int16_t>(imContig, &imOut, sigmas, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		PyWrapGaussian_run<uint32_t>(imContig, &imOut, sigmas, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		PyWrapGaussian_run<int32_t>(imContig, &imOut, sigmas, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		PyWrapGaussian_run<float>(imContig, &imOut, sigmas, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		PyWrapGaussian_run<double>(imContig, &imOut, sigmas, numIterations, device);
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

#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"

#include "PyKernel.h"


const char PyWrapMedianFilter::docString[] = "imageOut = HIP.MedianFilter(imageIn,kernel,[numIterations],[device])\n\n"\
"This will calculate the median for each neighborhood defined by the kernel.\n"\
"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"\
"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"\
"\t\thow to stride or jump to the next spatial block.\n"\
"\n"\
"\tkernel = This is a one to three dimensional array that will be used to determine neighborhood operations.\n"\
"\t\tIn this case, the positions in the kernel that do not equal zeros will be evaluated.\n"\
"\t\tIn other words, this can be viewed as a structuring element for the max neighborhood.\n"\
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
void PyWrapMedianFilter_run(const PyArrayObject* inIm, PyArrayObject** outIm, ImageContainer<float> kernel, int numIterations, int device)
{
	T* imageInPtr;
	T* imageOutPtr;

	ImageDimensions imageDims;
	Script::setupImagePointers(inIm, &imageInPtr, imageDims, outIm, &imageOutPtr);

	ImageContainer<T> imageIn(imageInPtr, imageDims);
	ImageContainer<T> imageOut(imageOutPtr, imageDims);

	medianFilter(imageIn, imageOut, kernel, numIterations, device);
}


PyObject* PyWrapMedianFilter::execute(PyObject* self, PyObject* args)
{
	PyObject* imIn;
	PyObject* inKern;

	int numIterations = 1;
	int device = -1;

	if ( !PyArg_ParseTuple(args, "O!O!|ii", &PyArray_Type, &imIn, &PyArray_Type, &inKern,
		&numIterations, &device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;
	if ( inKern == nullptr ) return nullptr;

	PyArrayObject* kernContig = (PyArrayObject*)PyArray_FROM_OTF(inKern, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

	ImageContainer<float> kernel = getKernel(kernContig);

	if ( kernel.getDims().getNumElements() == 0 )
	{
		kernel.clear();

		PyErr_SetString(PyExc_RuntimeError, "Unable to create kernel");
		return nullptr;
	}

	PyArrayObject* imOut = nullptr;

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		PyWrapMedianFilter_run<bool>(imContig, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		PyWrapMedianFilter_run<uint8_t>(imContig, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		PyWrapMedianFilter_run<uint16_t>(imContig, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		PyWrapMedianFilter_run<int16_t>(imContig, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		PyWrapMedianFilter_run<uint32_t>(imContig, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		PyWrapMedianFilter_run<int32_t>(imContig, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		PyWrapMedianFilter_run<float>(imContig, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		PyWrapMedianFilter_run<double>(imContig, &imOut, kernel, numIterations, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");

		Py_XDECREF(imContig);
		Py_XDECREF(kernContig);

		return nullptr;
	}

	Py_XDECREF(imContig);
	Py_XDECREF(kernContig);

	kernel.clear();

	return ((PyObject*)imOut);
}

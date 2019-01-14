#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

#include "PyKernel.h"


const char PyWrapClosure::docString[] = "imageOut = HIP.Closure(imageIn,kernel,[numIterations],[device])\n\n"\
"This kernel will dilate followed by an erosion.\n"\
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
void PyWrapClosure_run(const PyArrayObject* inIm, PyArrayObject** outIm, ImageView<float> kernel, int numIterations, int device)
{
	Script::DimInfo inInfo = Script::getDimInfo(inIm);
	ImageView<T> imageIn = Script::wrapInputImage<T>(inIm, inInfo);
	ImageView<T> imageOut = Script::createOutputImage<T>(outIm, inInfo);

	closure(imageIn, imageOut, kernel, numIterations, device);
}


PyObject* PyWrapClosure::execute(PyObject* self, PyObject* args)
{
	PyArrayObject* imIn;
	PyArrayObject* inKern;

	int numIterations = 1;
	int device = -1;

	if ( !PyArg_ParseTuple(args, "O!O!|ii", &PyArray_Type, &imIn, &PyArray_Type, &inKern,
						&numIterations, &device) )
		return nullptr;

	// TODO: These checks should be superfluous
	if ( imIn == nullptr ) return nullptr;
	if ( inKern == nullptr ) return nullptr;

	if ( !Script::ArrayInfo::isContiguous(imIn) )
	{
		PyErr_SetString(PyExc_RuntimeError, "Input image must be a contiguous numpy array!");
		return nullptr;
	}

	if ( !Script::ArrayInfo::isContiguous(inKern) )
	{
		PyErr_SetString(PyExc_RuntimeError, "Input kernel must be a contiguous numpy array!");
		return nullptr;
	}

	ImageOwner<float> kernel = getKernel(inKern);
	if ( kernel.getDims().getNumElements() == 0 )
	{
		PyErr_SetString(PyExc_RuntimeError, "Unable to create kernel");
		return nullptr;
	}

	ImageDimensions imageDims;
	PyArrayObject* imOut = nullptr;

	if ( PyArray_TYPE(imIn) == NPY_BOOL )
	{
		PyWrapClosure_run<bool>(imIn, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT8 )
	{
		PyWrapClosure_run<uint8_t>(imIn, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT16 )
	{
		PyWrapClosure_run<uint16_t>(imIn, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT16 )
	{
		PyWrapClosure_run<int16_t>(imIn, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT32 )
	{
		PyWrapClosure_run<uint32_t>(imIn, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT32 )
	{
		PyWrapClosure_run<int32_t>(imIn, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_FLOAT )
	{
		PyWrapClosure_run<float>(imIn, &imOut, kernel, numIterations, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_DOUBLE )
	{
		PyWrapClosure_run<double>(imIn, &imOut, kernel, numIterations, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");
		return nullptr;
	}

	return ((PyObject*)imOut);
}

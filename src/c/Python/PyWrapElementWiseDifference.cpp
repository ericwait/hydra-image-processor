#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

#include "PyKernel.h"

// TODO: Fixup documentation to clarify subtraction!!!
const char PyWrapElementWiseDifference::docString[] = "imageOut = HIP.ElementWiseDifference(imageIn1,imageIn2,[device])"\
"This subtracts the second array from the first, element by element (A-B).\n"\
"\timage1In = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"\
"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"\
"\t\thow to stride or jump to the next spatial block.\n"\
"\n"\
"\timage2In = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"\
"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"\
"\t\thow to stride or jump to the next spatial block.\n"\
"\n"\
"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"\
"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"\
"\t\tthe data across multiple devices.\n"\
"\n"\
"\timageOut = This will be an array of the same type and shape as the input array.";

template <typename T>
void PyWrapDiff_run(const PyArrayObject* inIm1, const PyArrayObject* inIm2, PyArrayObject** outIm, int device)
{
	Script::DimInfo inInfo1 = Script::getDimInfo(inIm1);
	Script::DimInfo inInfo2 = Script::getDimInfo(inIm1);
	Script::DimInfo outInfo = Script::maxDims(inInfo1, inInfo2);

	ImageView<T> image1In = Script::wrapInputImage<T>(inIm1, inInfo1);
	ImageView<T> image2In = Script::wrapInputImage<T>(inIm2, inInfo2);
	ImageView<T> imageOut = Script::createOutputImage<T>(outIm, outInfo);

	elementWiseDifference(image1In, image2In, imageOut, device);
}


PyObject* PyWrapElementWiseDifference::execute(PyObject*self, PyObject* args)
{
	int device = -1;

	PyArrayObject* imIn1;
	PyArrayObject* imIn2;

	if ( !PyArg_ParseTuple(args, "O!O!|i", &PyArray_Type, &imIn1, &PyArray_Type, &imIn2,
							&device) )
		return nullptr;

	// TODO: These checks should be superfluous
	if ( imIn1 == nullptr ) return nullptr;
	if ( imIn2 == nullptr ) return nullptr;

	if ( !Script::ArrayInfo::isContiguous(imIn1) )
	{
		PyErr_SetString(PyExc_RuntimeError, "Input image 1 must be a contiguous numpy array!");
		return nullptr;
	}

	if ( !Script::ArrayInfo::isContiguous(imIn2) )
	{
		PyErr_SetString(PyExc_RuntimeError, "Input kernel 2 must be a contiguous numpy array!");
		return nullptr;
	}

	PyArrayObject* imOut = nullptr;
	if ( PyArray_TYPE(imIn1) == NPY_BOOL )
	{
		PyWrapDiff_run<bool>(imIn1, imIn2, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn1) == NPY_UINT8 )
	{
		PyWrapDiff_run<uint8_t>(imIn1, imIn2, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn1) == NPY_UINT16 )
	{
		PyWrapDiff_run<uint16_t>(imIn1, imIn2, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn1) == NPY_INT16 )
	{
		PyWrapDiff_run<int16_t>(imIn1, imIn2, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn1) == NPY_UINT32 )
	{
		PyWrapDiff_run<uint32_t>(imIn1, imIn2, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn1) == NPY_INT32 )
	{
		PyWrapDiff_run<int32_t>(imIn1, imIn2, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn1) == NPY_FLOAT )
	{
		PyWrapDiff_run<float>(imIn1, imIn2, &imOut, device);
	}
	else if ( PyArray_TYPE(imIn1) == NPY_DOUBLE )
	{
		PyWrapDiff_run<double>(imIn1, imIn2, &imOut, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");
		return nullptr;
	}

	return ((PyObject*)imOut);
}

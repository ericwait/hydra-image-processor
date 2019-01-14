#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

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

template <typename T>
void PyWrapWienerFilter_run(const PyArrayObject* inIm, PyArrayObject** outIm, ImageView<float> kernel, double noiseVar, int device)
{
	Script::DimInfo inInfo = Script::getDimInfo(inIm);
	ImageView<T> imageIn = Script::wrapInputImage<T>(inIm, inInfo);
	ImageView<T> imageOut = Script::createOutputImage<T>(outIm, inInfo);

	wienerFilter(imageIn, imageOut, kernel, noiseVar, device);
}


PyObject* PyWrapWienerFilter::execute(PyObject* self, PyObject* args)
{
	PyArrayObject* imIn;
	PyArrayObject* inKern;

	double noiseVar = -1.0;
	int device = -1;

	if ( !PyArg_ParseTuple(args, "O!O!|di", &PyArray_Type, &imIn, &PyArray_Type, &inKern,
							&noiseVar, &device) )
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

	PyArrayObject* imOut = nullptr;
	if ( PyArray_TYPE(imIn) == NPY_BOOL )
	{
		PyWrapWienerFilter_run<bool>(imIn, &imOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT8 )
	{
		PyWrapWienerFilter_run<uint8_t>(imIn, &imOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT16 )
	{
		PyWrapWienerFilter_run<uint16_t>(imIn, &imOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT16 )
	{
		PyWrapWienerFilter_run<int16_t>(imIn, &imOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT32 )
	{
		PyWrapWienerFilter_run<uint32_t>(imIn, &imOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT32 )
	{
		PyWrapWienerFilter_run<int32_t>(imIn, &imOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_FLOAT )
	{
		PyWrapWienerFilter_run<float>(imIn, &imOut, kernel, noiseVar, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_DOUBLE )
	{
		PyWrapWienerFilter_run<double>(imIn, &imOut, kernel, noiseVar, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");
		return nullptr;
	}

	return ((PyObject*)imOut);
}

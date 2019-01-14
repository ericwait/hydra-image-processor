#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

#include "PyKernel.h"


const char PyWrapEntropyFilter::docString[] = "imageOut = HIP.EntropyFilter(imageIn,kernel,[device])\n\n"\
"This calculates the entropy within the neighborhood given by the kernel.\n"\
"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"\
"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"\
"\t\thow to stride or jump to the next spatial block.\n"\
"\n"\
"\tkernel = This is a one to three dimensional array that will be used to determine neighborhood operations.\n"\
"\t\tIn this case, the positions in the kernel that do not equal zeros will be evaluated.\n"\
"\t\tIn other words, this can be viewed as a structuring element for the max neighborhood.\n"\
"\n"\
"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"\
"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"\
"\t\tthe data across multiple devices.\n"\
"\n"\
"\timageOut = This will be an array of the same type and shape as the input array.\n";

template <typename InType, typename OutType>
void PyWrapEntropy_run(const PyArrayObject* inIm, PyArrayObject** outIm, ImageView<float> kernel, int device)
{
	Script::DimInfo inInfo = Script::getDimInfo(inIm);
	ImageView<InType> imageIn = Script::wrapInputImage<InType>(inIm, inInfo);
	ImageView<OutType> imageOut = Script::createOutputImage<OutType>(outIm, inInfo);

	entropyFilter(imageIn, imageOut, kernel, device);
}


PyObject* PyWrapEntropyFilter::execute(PyObject* self, PyObject* args)
{
	int device = -1;

	PyArrayObject* imIn;
	PyArrayObject* inKern;

	if ( !PyArg_ParseTuple(args, "O!O!|ii", &PyArray_Type, &imIn, &PyArray_Type, &inKern,
							&device) )
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
		PyWrapEntropy_run<bool,float>(imIn, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT8 )
	{
		PyWrapEntropy_run<uint8_t,float>(imIn, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT16 )
	{
		PyWrapEntropy_run<uint16_t,float>(imIn, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT16 )
	{
		PyWrapEntropy_run<int16_t,float>(imIn, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT32 )
	{
		PyWrapEntropy_run<uint32_t,float>(imIn, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT32 )
	{
		PyWrapEntropy_run<int32_t,float>(imIn, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_FLOAT )
	{
		PyWrapEntropy_run<float,float>(imIn, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_DOUBLE )
	{
		PyWrapEntropy_run<double,float>(imIn, &imOut, kernel, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");
		return nullptr;
	}

	return ((PyObject*)imOut);
}

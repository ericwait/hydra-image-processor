#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

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

template <typename InType, typename SumType>
void PyWrapSum_run(const PyArrayObject* inIm, PyObject** outSum, int device)
{
	Script::DimInfo inInfo = Script::getDimInfo(inIm);
	ImageView<InType> imageIn = Script::wrapInputImage<InType>(inIm, inInfo);

	SumType outVal = 0;
	sum(imageIn, outVal, device);

	*outSum = Script::Converter::fromNumeric<SumType>(outVal);
}


PyObject* PyWrapSum::execute(PyObject* self, PyObject* args)
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

	PyObject* outSum = nullptr;
	if ( PyArray_TYPE(imIn) == NPY_BOOL )
	{
		PyWrapSum_run<bool,uint64_t>(imIn, &outSum, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT8 )
	{
		PyWrapSum_run<uint8_t,uint64_t>(imIn, &outSum, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT16 )
	{
		PyWrapSum_run<uint16_t,uint64_t>(imIn, &outSum, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT16 )
	{
		PyWrapSum_run<int16_t,int64_t>(imIn, &outSum, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_UINT32 )
	{
		PyWrapSum_run<uint32_t,uint64_t>(imIn, &outSum, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_INT32 )
	{
		PyWrapSum_run<int32_t,int64_t>(imIn, &outSum, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_FLOAT )
	{
		PyWrapSum_run<float,double>(imIn, &outSum, device);
	}
	else if ( PyArray_TYPE(imIn) == NPY_DOUBLE )
	{
		PyWrapSum_run<double,double>(imIn, &outSum, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");
		return nullptr;
	}

	return outSum;
}

#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"

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
	InType* imageInPtr;

	ImageDimensions imageDims;
	Script::setupImagePointers(inIm, &imageInPtr, imageDims);

	ImageView<InType> imageIn(imageInPtr, imageDims);

	SumType outVal = 0;
	sum(imageIn, outVal, device);

	*outSum = Script::fromNumeric<SumType>(outVal);
}


PyObject* PyWrapSum::execute(PyObject* self, PyObject* args)
{
	PyObject* imIn;

	int device = -1;

	if ( !PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &imIn, &device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;

	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
	PyObject* outSum = nullptr;

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		PyWrapSum_run<bool,std::size_t>(imContig, &outSum, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		PyWrapSum_run<uint8_t,std::size_t>(imContig, &outSum, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		PyWrapSum_run<uint16_t,std::size_t>(imContig, &outSum, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		PyWrapSum_run<int16_t,long long>(imContig, &outSum, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		PyWrapSum_run<uint32_t,std::size_t>(imContig, &outSum, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		PyWrapSum_run<int32_t,long long>(imContig, &outSum, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		PyWrapSum_run<float,double>(imContig, &outSum, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		PyWrapSum_run<double,double>(imContig, &outSum, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");

		Py_XDECREF(imContig);

		return nullptr;
	}

	Py_XDECREF(imContig);

	return outSum;
}

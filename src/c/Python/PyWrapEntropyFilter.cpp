#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"

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
	InType* imageInPtr;
	OutType* imageOutPtr;

	ImageDimensions imageDims;
	Script::setupInputPointers(inIm, imageDims, &imageInPtr);
	Script::setupOutputPointers(outIm, imageDims, &imageOutPtr);

	ImageView<InType> imageIn(imageInPtr, imageDims);
	ImageView<OutType> imageOut(imageOutPtr, imageDims);

	entropyFilter(imageIn, imageOut, kernel, device);
}


PyObject* PyWrapEntropyFilter::execute(PyObject* self, PyObject* args)
{
	int device = -1;

	PyObject* imIn;
	PyObject* inKern;

	if ( !PyArg_ParseTuple(args, "O!O!|ii", &PyArray_Type, &imIn, &PyArray_Type, &inKern,
							&device) )
		return nullptr;

	if ( imIn == nullptr ) return nullptr;
	if ( inKern == nullptr ) return nullptr;

	PyArrayObject* kernContig = (PyArrayObject*)PyArray_FROM_OTF(inKern, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* imContig = (PyArrayObject*)PyArray_FROM_OTF(imIn, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

	ImageOwner<float> kernel = getKernel(kernContig);

	if ( kernel.getDims().getNumElements() == 0 )
	{
		Py_XDECREF(imContig);
		Py_XDECREF(kernContig);

		PyErr_SetString(PyExc_RuntimeError, "Unable to create kernel");
		return nullptr;
	}


	PyArrayObject* imOut = nullptr;

	if ( PyArray_TYPE(imContig) == NPY_BOOL )
	{
		PyWrapEntropy_run<bool,float>(imContig, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT8 )
	{
		PyWrapEntropy_run<uint8_t,float>(imContig, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT16 )
	{
		PyWrapEntropy_run<uint16_t,float>(imContig, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT16 )
	{
		PyWrapEntropy_run<int16_t,float>(imContig, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_UINT32 )
	{
		PyWrapEntropy_run<uint32_t,float>(imContig, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_INT32 )
	{
		PyWrapEntropy_run<int32_t,float>(imContig, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_FLOAT )
	{
		PyWrapEntropy_run<float,float>(imContig, &imOut, kernel, device);
	}
	else if ( PyArray_TYPE(imContig) == NPY_DOUBLE )
	{
		PyWrapEntropy_run<double,float>(imContig, &imOut, kernel, device);
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

	return ((PyObject*)imOut);
}

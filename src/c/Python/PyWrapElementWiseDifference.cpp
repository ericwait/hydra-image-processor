#include "PyWrapCommand.h"

#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"

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


PyObject* PyWrapElementWiseDifference::execute(PyObject*self, PyObject* args)
{
	int device = -1;

	PyObject* imIn1;
	PyObject* imIn2;

	if ( !PyArg_ParseTuple(args, "O!O!|i", &PyArray_Type, &imIn1, &PyArray_Type, &imIn2,
							&device) )
		return nullptr;

	if ( imIn1 == nullptr ) return nullptr;
	if ( imIn2 == nullptr ) return nullptr;

	PyArrayObject* im1Contig = (PyArrayObject*)PyArray_FROM_OTF(imIn1, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* im2Contig = (PyArrayObject*)PyArray_FROM_OTF(imIn2, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

	PyArrayObject* imOut = nullptr;

	ImageDimensions imageDims;
	if ( PyArray_TYPE(im1Contig) == NPY_BOOL )
	{
		bool* image1InPtr, *image2InPtr, *imageOutPtr;

		ImageDimensions image1Dims;
		setupInputPointers(im1Contig, image1Dims, &image1InPtr);
		ImageDimensions image2Dims;
		setupInputPointers(im2Contig, image2Dims, &image2InPtr);

		imageDims = ImageDimensions(Vec<size_t>::max(image1Dims.dims, image2Dims.dims), MAX(image1Dims.chan, image2Dims.chan), MAX(image1Dims.frame, image2Dims.frame));
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<bool> image1In(image1InPtr, image1Dims);
		ImageContainer<bool> image2In(image2InPtr, image2Dims);
		ImageContainer<bool> imageOut(imageOutPtr, imageDims);

		elementWiseDifference(image1In, image2In, imageOut, device);

	}
	else if ( PyArray_TYPE(im1Contig) == NPY_UINT8 )
	{
		unsigned char* image1InPtr, *image2InPtr, *imageOutPtr;

		ImageDimensions image1Dims;
		setupInputPointers(im1Contig, image1Dims, &image1InPtr);
		ImageDimensions image2Dims;
		setupInputPointers(im2Contig, image2Dims, &image2InPtr);

		imageDims = ImageDimensions(Vec<size_t>::max(image1Dims.dims, image2Dims.dims), MAX(image1Dims.chan, image2Dims.chan), MAX(image1Dims.frame, image2Dims.frame));
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<unsigned char> image1In(image1InPtr, image1Dims);
		ImageContainer<unsigned char> image2In(image2InPtr, image2Dims);
		ImageContainer<unsigned char> imageOut(imageOutPtr, imageDims);

		elementWiseDifference(image1In, image2In, imageOut, device);
	}
	else if ( PyArray_TYPE(im1Contig) == NPY_UINT16 )
	{
		unsigned short* image1InPtr, *image2InPtr, *imageOutPtr;

		ImageDimensions image1Dims;
		setupInputPointers(im1Contig, image1Dims, &image1InPtr);
		ImageDimensions image2Dims;
		setupInputPointers(im2Contig, image2Dims, &image2InPtr);

		imageDims = ImageDimensions(Vec<size_t>::max(image1Dims.dims, image2Dims.dims), MAX(image1Dims.chan, image2Dims.chan), MAX(image1Dims.frame, image2Dims.frame));
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<unsigned short> image1In(image1InPtr, image1Dims);
		ImageContainer<unsigned short> image2In(image2InPtr, image2Dims);
		ImageContainer<unsigned short> imageOut(imageOutPtr, imageDims);

		elementWiseDifference(image1In, image2In, imageOut, device);
	}
	else if ( PyArray_TYPE(im1Contig) == NPY_INT16 )
	{
		short* image1InPtr, *image2InPtr, *imageOutPtr;

		ImageDimensions image1Dims;
		setupInputPointers(im1Contig, image1Dims, &image1InPtr);
		ImageDimensions image2Dims;
		setupInputPointers(im2Contig, image2Dims, &image2InPtr);

		imageDims = ImageDimensions(Vec<size_t>::max(image1Dims.dims, image2Dims.dims), MAX(image1Dims.chan, image2Dims.chan), MAX(image1Dims.frame, image2Dims.frame));
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<short> image1In(image1InPtr, image1Dims);
		ImageContainer<short> image2In(image2InPtr, image2Dims);
		ImageContainer<short> imageOut(imageOutPtr, imageDims);

		elementWiseDifference(image1In, image2In, imageOut, device);
	}
	else if ( PyArray_TYPE(im1Contig) == NPY_UINT32 )
	{
		unsigned int* image1InPtr, *image2InPtr, *imageOutPtr;

		ImageDimensions image1Dims;
		setupInputPointers(im1Contig, image1Dims, &image1InPtr);
		ImageDimensions image2Dims;
		setupInputPointers(im2Contig, image2Dims, &image2InPtr);

		imageDims = ImageDimensions(Vec<size_t>::max(image1Dims.dims, image2Dims.dims), MAX(image1Dims.chan, image2Dims.chan), MAX(image1Dims.frame, image2Dims.frame));
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<unsigned int> image1In(image1InPtr, image1Dims);
		ImageContainer<unsigned int> image2In(image2InPtr, image2Dims);
		ImageContainer<unsigned int> imageOut(imageOutPtr, imageDims);

		elementWiseDifference(image1In, image2In, imageOut, device);
	}
	else if ( PyArray_TYPE(im1Contig) == NPY_INT32 )
	{
		int* image1InPtr, *image2InPtr, *imageOutPtr;

		ImageDimensions image1Dims;
		setupInputPointers(im1Contig, image1Dims, &image1InPtr);
		ImageDimensions image2Dims;
		setupInputPointers(im2Contig, image2Dims, &image2InPtr);

		imageDims = ImageDimensions(Vec<size_t>::max(image1Dims.dims, image2Dims.dims), MAX(image1Dims.chan, image2Dims.chan), MAX(image1Dims.frame, image2Dims.frame));
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<int> image1In(image1InPtr, image1Dims);
		ImageContainer<int> image2In(image2InPtr, image2Dims);
		ImageContainer<int> imageOut(imageOutPtr, imageDims);

		elementWiseDifference(image1In, image2In, imageOut, device);
	}
	else if ( PyArray_TYPE(im1Contig) == NPY_FLOAT )
	{
		float* image1InPtr, *image2InPtr, *imageOutPtr;

		ImageDimensions image1Dims;
		setupInputPointers(im1Contig, image1Dims, &image1InPtr);
		ImageDimensions image2Dims;
		setupInputPointers(im2Contig, image2Dims, &image2InPtr);

		imageDims = ImageDimensions(Vec<size_t>::max(image1Dims.dims, image2Dims.dims), MAX(image1Dims.chan, image2Dims.chan), MAX(image1Dims.frame, image2Dims.frame));
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<float> image1In(image1InPtr, image1Dims);
		ImageContainer<float> image2In(image2InPtr, image2Dims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		elementWiseDifference(image1In, image2In, imageOut, device);
	}
	else if ( PyArray_TYPE(im1Contig) == NPY_DOUBLE )
	{
		double* image1InPtr, *image2InPtr, *imageOutPtr;

		ImageDimensions image1Dims;
		setupInputPointers(im1Contig, image1Dims, &image1InPtr);
		ImageDimensions image2Dims;
		setupInputPointers(im2Contig, image2Dims, &image2InPtr);

		imageDims = ImageDimensions(Vec<size_t>::max(image1Dims.dims, image2Dims.dims), MAX(image1Dims.chan, image2Dims.chan), MAX(image1Dims.frame, image2Dims.frame));
		setupOutputPointers(&imOut, imageDims, &imageOutPtr);

		ImageContainer<double> image1In(image1InPtr, image1Dims);
		ImageContainer<double> image2In(image2InPtr, image2Dims);
		ImageContainer<double> imageOut(imageOutPtr, imageDims);

		elementWiseDifference(image1In, image2In, imageOut, device);
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Image type not supported.");

		Py_XDECREF(im1Contig);
		Py_XDECREF(im2Contig);
		Py_XDECREF(imOut);

		return nullptr;
	}

	Py_XDECREF(im1Contig);
	Py_XDECREF(im2Contig);

	return ((PyObject*)imOut);
}

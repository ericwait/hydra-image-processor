#pragma once
#include <string>
#include <cstring>
#include <algorithm>

#include "../Cuda/Vec.h"
#include "../Cuda/ImageDimensions.cuh"

#include "PyIncludes.h"


// Simple template-specialization map for C++ to mex types
template <typename T> struct TypeMap { static const NPY_TYPES npyType; };
template <> struct TypeMap<bool> { static const NPY_TYPES npyType = NPY_BOOL; };
template <> struct TypeMap<char> { static const NPY_TYPES npyType = NPY_INT8; };
template <> struct TypeMap<short> { static const NPY_TYPES npyType = NPY_INT16; };
template <> struct TypeMap<int> { static const NPY_TYPES npyType = NPY_INT32; };
template <> struct TypeMap<unsigned char> { static const NPY_TYPES npyType = NPY_UINT8; };
template <> struct TypeMap<unsigned short> { static const NPY_TYPES npyType = NPY_UINT16; };
template <> struct TypeMap<unsigned int> { static const NPY_TYPES npyType = NPY_UINT32; };
template <> struct TypeMap<float> { static const NPY_TYPES npyType = NPY_FLOAT; };
template <> struct TypeMap<double> { static const NPY_TYPES npyType = NPY_DOUBLE; };


void setupDims(PyArrayObject* im, ImageDimensions& dimsOut);
bool pyobjToVec(PyObject* list_array, Vec<double>& outVec);

// General array creation method
template <typename T>
PyArrayObject* createArray(int ndim, npy_intp* dims)
{
	return ((PyArrayObject*)PyArray_SimpleNew(ndim, dims, TypeMap<T>::npyType));
}

template <typename T>
void setupImagePointers(PyArrayObject* imageIn, T** image, ImageDimensions& dims, PyArrayObject** argOut = nullptr, T** imageOut = nullptr)
{
	setupInputPointers(imageIn, dims, image);
	if ( argOut != nullptr && imageOut != nullptr )
		setupOutputPointers(argOut, dims, imageOut);
}

template <typename T>
void setupInputPointers(PyArrayObject* imageIn, ImageDimensions& dims, T** image)
{
	setupDims(imageIn, dims);
	*image = (T*)PyArray_DATA(imageIn);
}

template <typename T>
void setupOutputPointers(PyArrayObject** imageOut, ImageDimensions& dims, T** image)
{
	npy_intp pyDims[5];
	for ( int i = 0; i < 3; ++i )
		pyDims[i] = dims.dims.e[i];

	pyDims[3] = dims.chan;
	pyDims[4] = dims.frame;

	*imageOut = createArray<T>(5, pyDims);
	*image = (T*)PyArray_DATA(*imageOut);

	std::memset(*image, 0, sizeof(T)*dims.getNumElements());
}


#include "PyWrapDef.h"
#include "../WrapCmds/CommandList.h"

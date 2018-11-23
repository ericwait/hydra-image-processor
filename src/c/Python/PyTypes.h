#pragma once

#include <Python.h>

// Make sure that Numpy symbols don't get re-imported in multiple compilation units
#ifndef NUMPY_IMPORT_MODULE
#define NO_IMPORT_ARRAY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL HIP_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <py3c.h>
#include <cstddef>

namespace Script
{
	typedef npy_intp		DimType;
	typedef PyObject		ObjectType;
	typedef PyArrayObject	ArrayType;

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

	template <typename T> ArrayType* createArray(int ndim, DimType* dims)
	{
		return ((ArrayType*) PyArray_SimpleNew(ndim, dims, TypeMap<T>::npyType));
	}

	// TODO: Figure out if we can do this without the const-casting?
	namespace ArrayInfo
	{
		inline std::size_t	getNDims(const ArrayType* im) { return PyArray_NDIM(im); }
		inline DimType* getDims(const ArrayType* im) { return PyArray_DIMS(const_cast<ArrayType*>(im)); }

		template <typename T>
		T* getData(const ArrayType* im) { return (T*) PyArray_DATA(const_cast<ArrayType*>(im)); }
	};

	// Some Python-specific converters
	bool pyobjToVec(ObjectType* list_array, Vec<double>& outVec);

	bool pylistToVec(ObjectType* list, Vec<double>& outVec);
	bool pyarrayToVec(ArrayType* ar, Vec<double>& outVec);
};

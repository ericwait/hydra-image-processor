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
#include <cstdint>

#define CHECK_ARRAY_FLAGS(IM,FLAG) ((PyArray_FLAGS(IM) & (FLAG)) == (FLAG))

namespace Script
{
	typedef npy_intp		DimType;
	typedef PyObject		ObjectType;
	typedef PyArrayObject	ArrayType;

	// Simple template-specialization map for C++ to Python types
	BEGIN_TYPE_MAP(NPY_TYPES)
		TYPE_MAPPING(bool, NPY_BOOL)
		TYPE_MAPPING(int8_t, NPY_INT8)
		TYPE_MAPPING(int16_t, NPY_INT16)
		TYPE_MAPPING(int32_t, NPY_INT32)
		TYPE_MAPPING(uint8_t, NPY_UINT8)
		TYPE_MAPPING(uint16_t, NPY_UINT16)
		TYPE_MAPPING(uint32_t, NPY_UINT32)
		TYPE_MAPPING(float, NPY_FLOAT)
		TYPE_MAPPING(double, NPY_DOUBLE)
	END_TYPE_MAP(NPY_TYPES)


	inline std::vector<DimType> arrayDims(const DimInfo& info)
	{
		if ( info.columnMajor )
			return std::vector<DimType>(info.dims.begin(), info.dims.end());
		else
			return std::vector<DimType>(info.dims.rbegin(), info.dims.rend());
	}

	template <typename T> ArrayType* createArray(const Script::DimInfo& info)
	{
		// Returns reverse-ordered dimensions in case of row-major array
		std::vector<DimType> dims = Script::arrayDims(info);
		return ((ArrayType*) PyArray_EMPTY(((int)dims.size()), dims.data(), ID_FROM_TYPE(T), ((int)info.columnMajor)));
	}

	// TODO: Figure out if we can do this without the const-casting?
	namespace ArrayInfo
	{
		inline bool isColumnMajor(const ArrayType* im) { return (CHECK_ARRAY_FLAGS(im,NPY_ARRAY_FARRAY_RO) && !CHECK_ARRAY_FLAGS(im, NPY_ARRAY_C_CONTIGUOUS)); }
		inline bool isContiguous(const ArrayType* im) { return (CHECK_ARRAY_FLAGS(im, NPY_ARRAY_CARRAY_RO) || CHECK_ARRAY_FLAGS(im, NPY_ARRAY_FARRAY_RO)) ; }

		inline std::size_t getNDims(const ArrayType* im) { return PyArray_NDIM(im); }
		inline DimType getDim(const ArrayType* im, int idim) { return PyArray_DIM(im, idim); }

		template <typename T>
		T* getData(const ArrayType* im) { return (T*)PyArray_DATA(const_cast<ArrayType*>(im)); }
	};

	// Some Python-specific converters
	bool pyobjToVec(ObjectType* list_array, Vec<double>& outVec);

	bool pylistToVec(ObjectType* list, Vec<double>& outVec);
	bool pyarrayToVec(ArrayType* ar, Vec<double>& outVec);

	template <typename T> PyObject* fromNumeric(T val){ static_assert(!std::is_same<T,T>::value, "Python type converion not implemented"); return nullptr; }
	template <> inline PyObject* fromNumeric(bool val) { return PyBool_FromLong(val); }
	template <> inline PyObject* fromNumeric(uint8_t val) { return PyLong_FromLong(val); }
	template <> inline PyObject* fromNumeric(uint16_t val) { return PyLong_FromLong(val); }
	template <> inline PyObject* fromNumeric(int16_t val) { return PyLong_FromLong(val); }
	template <> inline PyObject* fromNumeric(uint32_t val) { return PyLong_FromLong(val); }
	template <> inline PyObject* fromNumeric(int32_t val) { return PyLong_FromLong(val); }
	template <> inline PyObject* fromNumeric(float val) { return PyFloat_FromDouble(val); }
	template <> inline PyObject* fromNumeric(double val) { return PyFloat_FromDouble(val); }
    template <> inline PyObject* fromNumeric(long val) { return PyLong_FromLong(val); }
    template <> inline PyObject* fromNumeric(unsigned long val) { return PyLong_FromUnsignedLong(val); }
    template <> inline PyObject* fromNumeric(long long val) { return PyLong_FromLongLong(val); }
	template <> inline PyObject* fromNumeric(unsigned long long val) { return PyLong_FromUnsignedLongLong(val); }
};

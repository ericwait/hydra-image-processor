#pragma once

#include "MexIncludes.h"

#include <cstddef>
#include <cstdint>

namespace Script
{
	typedef mwSize	DimType;
	typedef mxArray ArrayType;
	typedef mxArray ObjectType;

	// Simple template-specialization map for C++ to mex types
	BEGIN_TYPE_MAP(mxClassID,Matlab)
		TYPE_MAPPING(bool, mxLOGICAL_CLASS)
		TYPE_MAPPING(int8_t, mxINT8_CLASS)
		TYPE_MAPPING(int16_t, mxINT16_CLASS)
		TYPE_MAPPING(int32_t, mxINT32_CLASS)
		TYPE_MAPPING(uint8_t, mxUINT8_CLASS)
		TYPE_MAPPING(uint16_t, mxUINT16_CLASS)
		TYPE_MAPPING(uint32_t, mxUINT32_CLASS)
		TYPE_MAPPING(float, mxSINGLE_CLASS)
		TYPE_MAPPING(double, mxDOUBLE_CLASS)
	END_TYPE_MAP(mxClassID)


	template <typename... Args>
	inline void writeMsg(const char* fmt, Args&&... args)
	{
		mexPrintf(fmt, std::forward<Args>(args)...);
	}

	template <typename... Args>
	inline void errorMsg(const char* fmt, Args&&... args)
	{
		mexErrMsgTxt(formatMsg(fmt, std::forward<Args>(args)...).c_str());
	}


	namespace Array
	{
		inline bool isColumnMajor(const ArrayType* im){return true;}
		inline bool isContiguous(const ArrayType* im){return true;}

		inline IdType getType(const ArrayType* im) { return mxGetClassID(im); }
		inline std::size_t getNDims(const ArrayType* im){ return mxGetNumberOfDimensions(im); }
		inline DimType getDim(const ArrayType* im, int idim){ return mxGetDimensions(im)[idim]; }

		template <typename T>
		T* getData(const ArrayType* im) { return (T*)mxGetData(im); }


		// Helper functions for array allocation
		template <typename T>
		ArrayType* create(const DimInfo& info)
		{
			return mxCreateNumericArray(info.dims.size(), info.dims.data(), ID_FROM_TYPE(T), mxREAL);
		}

		template <>
		inline ArrayType* create<bool>(const DimInfo& info)
		{
			return mxCreateLogicalArray(info.dims.size(), info.dims.data());
		}
	};


	inline bool isEmpty(const ObjectType* mexArray) { return mxIsEmpty(mexArray); }


	// Minimal wrapper around script structure types
	namespace Struct
	{
		inline const ObjectType* getVal(ObjectType* structPtr, std::size_t idx, const char* field)
		{
			return mxGetField(structPtr, idx, field);
		}

		inline void setVal(ObjectType* structPtr, std::size_t idx, const char* field, ObjectType* val)
		{
			mxSetField(structPtr, idx, field, val);
		}

		// Wrapper around script structure-array creation/access
		inline ObjectType* create(std::size_t size, const std::vector<const char*>& fields)
		{
			const char** fieldData = const_cast<const char**>(fields.data());
			return mxCreateStructMatrix(size, 1, (int)fields.size(), fieldData);
		}
	};

	// Helper Unique pointer type that frees memory with mxFree
	template <typename T>
	using mx_unique_ptr = std::unique_ptr<T, decltype(&mxFree)>;

	template <typename T>
	static mx_unique_ptr<T> make_mx_unique(T* ptr)
	{
		return mx_unique_ptr<T>(ptr, mxFree);
	}

	struct Converter: public ConvertErrors
	{
	private:
		template <typename OutT, typename InT>
		inline static void copyConvertArray(OutT* outPtr, const InT* inPtr, std::size_t length)
		{
			for ( std::size_t i = 0; i < length; ++i )
				outPtr[i] = static_cast<OutT>(inPtr[i]);
		}

		template <typename T>
		static void copyConvertArray(T* outPtr, const T* inPtr, std::size_t length)
		{
			std::memcpy(outPtr, inPtr, length*sizeof(T));
		}

		// Typed-array copy conversion
		template <typename T>
		static void arrayCopy(T* outPtr, const ArrayType* mexArray)
		{
			std::size_t array_size = mxGetNumberOfElements(mexArray);

			Script::IdType type = Array::getType(mexArray);
			if ( type == ID_FROM_TYPE(bool) )
				copyConvertArray(outPtr, Array::getData<bool>(mexArray), array_size);
			else if ( type == ID_FROM_TYPE(uint8_t) )
				copyConvertArray(outPtr, Array::getData<uint8_t>(mexArray), array_size);
			else if ( type == ID_FROM_TYPE(uint16_t) )
				copyConvertArray(outPtr, Array::getData<uint16_t>(mexArray), array_size);
			else if ( type == ID_FROM_TYPE(int16_t) )
				copyConvertArray(outPtr, Array::getData<int16_t>(mexArray), array_size);
			else if ( type == ID_FROM_TYPE(uint32_t) )
				copyConvertArray(outPtr, Array::getData<uint32_t>(mexArray), array_size);
			else if ( type == ID_FROM_TYPE(int32_t) )
				copyConvertArray(outPtr, Array::getData<int32_t>(mexArray), array_size);
			else if ( type == ID_FROM_TYPE(float) )
				copyConvertArray(outPtr, Array::getData<float>(mexArray), array_size);
			else if ( type == ID_FROM_TYPE(double) )
				copyConvertArray(outPtr, Array::getData<double>(mexArray), array_size);
			else
				throw ArrayTypeError("Unsupported matrix type: %x", type);
		}


	public:

		////////////
		// Basic scalar <-> Matlab conversions
		template <typename T, ENABLE_CHK(IS_BOOL(T))>
		inline static ObjectType* fromNumeric(T val) { return mxCreateLogicalScalar(val); }

		// Always return double scalar for non-logical types
		template <typename T, ENABLE_CHK(NUMERIC_NONBOOL(T))>
		inline static ObjectType* fromNumeric(T val) { return mxCreateDoubleScalar(static_cast<double>(val)); }


		template <typename T, ENABLE_CHK(IS_BOOL(T))>
		inline static T toNumeric(const ObjectType* mexArray)
		{
			if ( !mxIsLogicalScalar(mexArray) )
				throw ScalarConvertError("Expected scalar boolean value");

			return mxGetLogicals(mexArray)[0];
		}

		template <typename T, ENABLE_CHK(NUMERIC_NONBOOL(T))>
		inline static T toNumeric(const ObjectType* mexArray)
		{
			if ( !mxIsScalar(mexArray) || !mxIsDouble(mexArray) )
				throw ScalarConvertError("Expected scalar double value");

			return static_cast<T>(mxGetScalar(mexArray));
		}


		// Matlab <-> string conversion
		inline static ObjectType* fromString(const char* str)
		{
			return mxCreateString(str);
		}

		inline static ObjectType* fromString(const std::string& str)
		{
			return fromString(str.c_str());
		}

		inline static std::string toString(const ObjectType* mexArray)
		{
			if ( !mxIsChar(mexArray) )
				ArgConvertError(nullptr, "Expected a string argument");

			mx_unique_ptr<char> strPtr = make_mx_unique(mxArrayToUTF8String(mexArray));
			if ( !strPtr )
				ArgConvertError(nullptr, "Invalid string argument");

			return std::string(strPtr.get());
		}


		////////////
		// Vec <-> Matlab conversions
		template <typename T, ENABLE_CHK(IS_BOOL(T))>
		inline static ObjectType* fromVec(const Vec<T>& vec)
		{
			ObjectType* mexVec = mxCreateLogicalMatrix(1,3);
			mxLogical* data = (mxLogical*)mxGetData(mexVec);

			for ( int i=0; i < 3; ++i )
				data[i] = static_cast<mxLogical>(vec.e[i]);

			return mexVec;
		}

		template <typename T, ENABLE_CHK(NUMERIC_NONBOOL(T))>
		inline static ObjectType* fromVec(const Vec<T>& vec)
		{
			ObjectType* mexVec = mxCreateDoubleMatrix(1,3, mxREAL);
			double* data = mxGetPr(mexVec);

			for ( int i=0; i < 3; ++i )
				data[i] = static_cast<double>(vec.e[i]);

			return mexVec;
		}


		template <typename T, ENABLE_CHK(IS_BOOL(T))>
		inline static Vec<T> toVec(const ObjectType* mexArray)
		{
			if ( !mxIsLogical(mexArray) || mxGetNumberOfElements(mexArray) != 3 )
				throw VectorConvertError("Expected a 3-element logical vector");

			mxLogical* data = Script::Array::getData<mxLogical>(mexArray);

			Vec<T> outVec;
			for ( int i=0; i < 3; ++i )
				outVec.e[i] = static_cast<T>(data[i]);

			return outVec;
		}

		template <typename T, ENABLE_CHK(NUMERIC_NONBOOL(T))>
		inline static Vec<T> toVec(const ObjectType* mexArray)
		{
			if ( !mxIsDouble(mexArray) || mxGetNumberOfElements(mexArray) != 3 )
				throw VectorConvertError("Expected a 3-element double vector");

			double* data = Script::Array::getData<double>(mexArray);

			Vec<T> outVec;
			for ( int i=0; i < 3; ++i )
				outVec.e[i] = static_cast<T>(data[i]);

			return outVec;
		}

		////////////
		// Matlab -> ImageView conversion
		// NOTE: Unlike most other conversions this is a pointer copy not a deep copy
		//  Thus the underlying data types must match EXACTLY
		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static ImageView<T> toImage(const ArrayType* mexArray)
		{
			if ( !mxIsNumeric(mexArray) || !mxIsLogical(mexArray) )
				throw ImageConvertError("Expected a numeric or logical matrix");

			Script::IdType type = Script::Array::getType(mexArray);
			if ( ID_FROM_TYPE(T) != type )
				throw ImageConvertError("Expected array of type: %s", NAME_FROM_TYPE(T));

			Script::DimInfo info = Script::getDimInfo(mexArray);
			return ImageView<T>(Script::Array::getData<T>(mexArray), Script::makeImageDims(info));
		}


		// Matlab <-> ImageOwner conversion
		// NOTE: This is a deep-copy and conversion of the array data
		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static ImageOwner<T> toImageCopy(const ArrayType* mexArray)
		{
			if ( !mxIsNumeric(mexArray) || !mxIsLogical(mexArray) )
				throw ImageConvertError("Expected a numeric or logical matrix");

			Script::DimInfo info = Script::getDimInfo(mexArray);
			if ( !info.contiguous )
				throw ImageConvertError("Only contiguous numpy arrays are supported");

			ImageDimensions inDims = Script::makeImageDims(info);
			ImageOwner<T> outIm(inDims);

			arrayCopy(outIm.getPtr(), mexArray);
			return outIm;
		}

	};
};

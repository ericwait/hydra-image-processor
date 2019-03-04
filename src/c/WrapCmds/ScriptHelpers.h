#pragma once

#include "../Cuda/Vec.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

#include <cstddef>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>


#undef snprintf

#ifdef _WIN32
 #define SNPRINTF std::snprintf
#else
 #define SNPRINTF std::snprintf
#endif


#define BEGIN_TYPE_MAP(EnumType,ScriptEngine)		\
	typedef EnumType IdType;						\
	template <typename T> struct TypeToIdMap		\
	{static_assert(!std::is_same<T,T>::value, "HIP_COMPILE: No " #ScriptEngine " type mapping specified for T");};	\
	template <typename T> struct TypeNameMap		\
	{static_assert(!std::is_same<T,T>::value, "HIP_COMPILE: No " #ScriptEngine " type mapping specified for T");};

#define TYPE_MAPPING(Type,TypeID)															\
	template <> struct TypeToIdMap<Type> {static constexpr const IdType typeId = TypeID;};	\
	template <> struct TypeNameMap<Type> {static constexpr const char* name(){ return #Type;}};

#define END_TYPE_MAP(EnumType)


#define NAME_FROM_TYPE(Type) TypeNameMap<Type>::name()
#define ID_FROM_TYPE(Type) TypeToIdMap<Type>::typeId

namespace Script
{
	struct DimInfo
	{
		bool contiguous;
		bool columnMajor;
		std::vector<std::size_t> dims;
	};

	// Generic script conversion errors used by engine-specific converters
	class ConvertErrors
	{
	public:
		// Conversion exception types
		class ArgConvertError: public std::runtime_error
		{
			template <typename... Args>
			static std::unique_ptr<char[]> make_msg(const char* fmt, Args... args)
			{
				size_t size = SNPRINTF(nullptr, 0, fmt, args...);

				std::unique_ptr<char[]> msgPtr(new char[size+1]);
				SNPRINTF(msgPtr.get(), size, fmt, args...);
				return msgPtr;
			}

			static std::unique_ptr<char[]> make_msg(const char* fmt)
			{
				size_t size = std::strlen(fmt);

				std::unique_ptr<char[]> msgPtr(new char[size+1]);
				std::strncpy(msgPtr.get(), fmt, size);
				return msgPtr;
			}

		public:
			template <typename... Args>
			ArgConvertError(const char* argName, const char* fmt, Args... args)
				: std::runtime_error(make_msg(fmt, args...).get()), argName(argName)
			{}

			const char* getArgName() const { return argName; }
			void setArgName(const char* name) { argName = name; }

		private:
			ArgConvertError() = delete;

		private:
			const char* argName;
		};

		class ScalarConvertError: public ArgConvertError
		{
		public:
			template <typename... Args>
			ScalarConvertError(const char* fmt, Args... args)
				: ArgConvertError(nullptr, fmt, args...) {}
		};

		class VectorConvertError: public ArgConvertError
		{
		public:
			template <typename... Args>
			VectorConvertError(const char* fmt, Args... args)
				: ArgConvertError(nullptr, fmt, args...) {}
		};

		class ImageConvertError: public ArgConvertError
		{
		public:
			template <typename... Args>
			ImageConvertError(const char* fmt, Args... args)
				: ArgConvertError(nullptr, fmt, args...) {}
		};

		class ArrayTypeError: public ArgConvertError
		{
		public:
			template <typename... Args>
			ArrayTypeError(const char* fmt, Args... args)
				: ArgConvertError(nullptr, fmt, args...) {}
		};
	};
};


#if defined(PY_BUILD)
 #include "PyTypes.h"
#elif defined(MEX_BUILD)
 #include "MexTypes.h"
#endif

namespace Script
{
	// DimInfo handling functions
	inline DimInfo getDimInfo(const ArrayType* im)
	{
		DimInfo info;
		info.contiguous = ArrayInfo::isContiguous(im);
		info.columnMajor = ArrayInfo::isColumnMajor(im);

		std::size_t ndims = ArrayInfo::getNDims(im);
		info.dims.resize(ndims);

		// Load dimensions in reverse order if row-major
		int offset = (info.columnMajor) ? (0) : ((int)ndims-1);
		int m = (info.columnMajor) ? (1) : (-1);

		for ( int i=0; i < ndims; ++i )
			info.dims[i] = ArrayInfo::getDim(im, m*i + offset);

		return info;
	}

	inline DimInfo maxDims(const DimInfo& infoA, const DimInfo& infoB)
	{
		DimInfo outInfo = (infoA.dims.size() > infoB.dims.size()) ? infoA : infoB;

		std::size_t sharedSize = std::min(infoA.dims.size(), infoB.dims.size());
		for ( int i=0; i < sharedSize; ++i )
			outInfo.dims[i] = std::max(infoA.dims[i], infoB.dims[i]);

		return outInfo;
	}

	inline ImageDimensions makeImageDims(const DimInfo& info)
	{
		ImageDimensions dimsOut(Vec<std::size_t>(1), 1, 1);

		int nsdims = std::min<int>(3, (int)info.dims.size());
		for ( int i=0; i < nsdims; ++i )
			dimsOut.dims.e[i] = info.dims[i];

		if ( info.dims.size() >= 4 )
			dimsOut.chan = (unsigned int)info.dims[3];

		if ( info.dims.size() >= 5 )
			dimsOut.frame = (unsigned int)info.dims[4];

		return dimsOut;
	}


	// Script image creation wrappers
	template <typename T>
	ImageView<T> wrapInputImage(const ArrayType* imageIn, const DimInfo& info)
	{
		return ImageView<T>(ArrayInfo::getData<T>(imageIn), makeImageDims(info));
	}

	template <typename T>
	ImageView<T> createOutputImage(ArrayType** imageOut, const DimInfo& info)
	{
		*imageOut = createArray<T>(info);
		return wrapInputImage<T>(*imageOut, info);
	}
};

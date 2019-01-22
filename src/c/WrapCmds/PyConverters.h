#include "../Cuda/ImageContainer.h"

#include "mph/tuple_helpers.h"
#include "mph/qualifier_helpers.h"

#include <tuple>
#include <type_traits>

#define SCR_PARAMS(...) Script::ArgParser<__VA_ARGS__>

#define SCR_OUTPUT(TypeMacro) Script::OutParam<TypeMacro>
#define SCR_INPUT(TypeMacro) Script::InParam<TypeMacro>
#define SCR_OPTIONAL(TypeMacro, DefVal) Script::OptParam<TypeMacro>

#define SCR_IMAGE_CONVERT(VarName, VarType) Script::Image<VarType>
//#define SCR_IMAGE_REQUIRE(VarName, VarType)
#define SCR_IMAGE_DYNAMIC(VarName) Script::Image<Script::DeferredType>

#define SCR_SCALAR(VarName, VarType) Script::Scalar<VarType>
#define SCR_SCALAR_DYNAMIC(VarName) Script::Scalar<Script::DeferredType>

#define SCR_VECTOR(VarName, VarType) Script::Vector<VarType>
#define SCR_VECTOR_DYNAMIC(VarName) Script::Vector<Script::DeferredType>

namespace Script
{
	/////////////////////////
	// force_const_t -
	//   Tear away all pointers on type and set const on the underlying data type
	//   then add all the pointers back (e.g. int** -> const int**)
	/////////////////////////
	template <typename T>
	using force_const_t = mph::data_type_tfm<std::add_const, T>;


	/////////////////////////
	// remove_all_qualifiers_t -
	//   Recursively remove type qualifiers (e.g. const int * const * const -> int**)
	/////////////////////////
	template <typename T>
	using remove_all_qualifiers_t = mph::full_type_tfm_t<std::remove_cv, T>;
	
	
	// strip_outer - Remove one layer of nested types (used to remove In/Out/OptParam qualifier types)
	template <typename T>
	struct strip_outer {};

	template <template<typename> class T, typename U>
	struct strip_outer<T<U>>
	{
		using type = U;
	};


	// Type definitions
	using ScriptObjPtr = Script::ObjectType*;
	using ScriptArrayPtr = Script::ArrayType*;

	struct DeferredType {};

	template <typename T>
	struct Scalar
	{
		using Type = T;
		using ConcreteType = T;
	};

	template <>
	struct Scalar<DeferredType>
	{
		using Type = DeferredType;
		using ConcreteType = ScriptObjPtr;
	};

	template <typename T>
	struct Vector
	{
		using Type = T;
		using ConcreteType = Vec<T>;
	};

	template <>
	struct Vector<DeferredType>
	{
		using Type = DeferredType;
		using ConcreteType = ScriptObjPtr;
	};

	template <typename T>
	struct Image
	{
		using Type = T;
		using ConcreteType = ImageContainer<T>;
	};

	template <>
	struct Image<DeferredType>
	{
		using Type = DeferredType;
		using ConcreteType = ScriptArrayPtr;
	};


	// Type traits for concrete/deferred io
	template <typename T>
	struct is_deferred
	{
		static constexpr bool value = std::is_same<T, DeferredType>::value;
	};

	template <template<typename> class T, typename U>
	struct is_deferred<T<U>>
	{
		static constexpr bool value = is_deferred<U>::value;
	};

	// Check for image types
	template <typename T>
	struct is_image
	{
		static constexpr bool value = false;
	};

	template <template<typename> class T, typename U>
	struct is_image<T<U>>
	{
		static constexpr bool value = (std::is_same<Image<U>, T<U>>::value || is_image<U>::value);
	};

	// Convenience type for script inputs (e.g. const PyObject*)
	using const_script_in_t = force_const_t<ScriptObjPtr>;

	template <typename T>
	struct PyParseType
	{
		using type = const_script_in_t;
	};

	template <typename T>
	struct force_use_type
	{
		template <typename U>
		struct tfm
		{ using type = T; };
	};


	// Helper for python arg expanders
	template <typename T>
	struct ParserArg{};

	template <template<typename> class T, typename C>
	struct ParserArg<T<C>>
	{
		using ArgTuple = std::tuple<typename std::add_pointer<const_script_in_t>::type>;

		static ArgTuple argTuple(const_script_in_t& arg)
		{
			return std::make_tuple(&arg);
		}

		// TODO: Move to compile-time strings (maybe library)
		static const char* argString()
		{
			static const char parse_type[] = "O";
			return parse_type;
		};
	};

	template <typename C>
	struct ParserArg<Image<C>>
	{
		using ArgTuple = std::tuple<PyTypeObject*,
			typename std::add_pointer<const_script_in_t>::type>;

		static ArgTuple argTuple(const_script_in_t& arg)
		{
			return std::make_tuple(&PyArray_Type, &arg);
		}

		static const char* argString()
		{
			static const char parse_type[] = "O!";
			return parse_type;
		};
	};


	// Helper defines for python base-type conversion checks
	#define CHECK_PY_INT_CVT(OutType, OutPtr, InPtr) \
			if (PyLong_Check((InPtr))) {(*(OutPtr)) = PyLong_As##OutType((InPtr));} \
			else {return false;} \
			return true;

	#define CHECK_PY_FLOAT_CVT(OutPtr, InPtr) \
			if (PyLong_Check((InPtr))) {(*(OutPtr)) = PyLong_AsDouble((InPtr));} \
			else if (PyFloat_Check((InPtr))) {(*(OutPtr)) = PyFloat_AsDouble((InPtr));} \
			else {return false;} \
			return true;

	// Script-to-concrete input converter
	struct ScriptInConvert
	{
	public:
		template <typename T>
		static void convert(T& out, const PyObject* inPtr)
		{
			// NOTE: if inPtr is nullptr then this is presumed to be optional
			if ( inPtr == nullptr )
				return;

			convert_impl(out, inPtr);
		}

	private:
		// Have to check python object types when converting individual numbers
		static bool pyNumericConvert(long* out, PyObject* in) { CHECK_PY_INT_CVT(Long, out, in); }
		static bool pyNumericConvert(unsigned long* out, PyObject* in) { CHECK_PY_INT_CVT(UnsignedLong, out, in); }
		static bool pyNumericConvert(double* out, PyObject* in) { CHECK_PY_FLOAT_CVT(out, in); }

		// Copy-converter for python arrays
		// NOTE: This doesn't account for stride differences and should only be used for
		//   1-D numpy arrays
		template <typename OutType, typename InType>
		static void convertArray(OutType* outPtr, const InType* inPtr, std::size_t length)
		{
			for ( std::size_t i = 0; i < length; ++i )
				outPtr[i] = static_cast<OutType>(inPtr[i]);
		}

		// Specialization runs a simple memcpy
		template <typename T>
		static void convertArray(T* outPtr, const T* inPtr, std::size_t length)
		{
			std::memcpy(outPtr, inPtr, length*sizeof(T));
		}


		template <typename T>
		static void pyArrayCopyConvert(T* outPtr, PyArrayObject* arrPtr)
		{
			// TODO: Check for contiguous
			std::size_t array_size = PyArray_SIZE(arrPtr);

			void* data = PyArray_DATA(arrPtr);
			Script::IdType type = Script::ArrayInfo::getType(arrPtr);
			if ( type == NPY_UINT8 )
				convertArray(outPtr, reinterpret_cast<uint8_t*>(data), array_size);
			else if ( type == NPY_UINT16 )
				convertArray(outPtr, reinterpret_cast<uint16_t*>(data), array_size);
			else if ( type == NPY_INT16 )
				convertArray(outPtr, reinterpret_cast<int16_t*>(data), array_size);
			else if ( type == NPY_UINT32 )
				convertArray(outPtr, reinterpret_cast<uint32_t*>(data), array_size);
			else if ( type == NPY_INT32 )
				convertArray(outPtr, reinterpret_cast<int32_t*>(data), array_size);
			else if ( type == NPY_FLOAT )
				convertArray(outPtr, reinterpret_cast<float*>(data), array_size);
			else if ( type == NPY_DOUBLE )
				convertArray(outPtr, reinterpret_cast<double*>(data), array_size);
			else
			{
				// TODO: Throw type error
			}
		}


		// Helpers for Vec<T> conversion
		template <typename T>
		static void pyListToVec(Vec<T>& out, const PyObject* inPtr)
		{
			PyObject* list = const_cast<PyObject*>(inPtr);
			Py_ssize_t list_size = PyList_Size(list);
			if ( list_size != 3 )
				return; // TODO: Throw type error

			for ( int i=0; i < list_size; ++i )
			{
				PyObject* item = PyList_GetItem(list, i);
				convert_impl(out.e[i], item);
			}
		}

		template <typename T>
		static void pyTupleToVec(Vec<T>& out, const PyObject* inPtr)
		{
			PyObject* tuple = const_cast<PyObject*>(inPtr);
			Py_ssize_t tuple_size = PyTuple_Size(tuple);
			if ( tuple_size != 3 )
				return; // TODO: Throw type error

			for ( int i=0; i < tuple_size; ++i )
			{
				PyObject* item = PyTuple_GetItem(tuple, i);
				convert_impl(out.e[i], item);
			}
		}

		template <typename T>
		static void pyArrayToVec(Vec<T>& out, const PyObject* inPtr)
		{
			PyArrayObject* arrPtr = const_cast<PyArrayObject*>(reinterpret_cast<const PyArrayObject*>(inPtr));
			size_t ndim = Script::ArrayInfo::getNDims(arrPtr);
			if ( ndim > 1 )
				return; // TODO: Throw type error!

			Script::DimType* DIMS = Script::ArrayInfo::getDims(arrPtr);
			if ( DIMS[0] != 3 )
				return; // TODO: Throw type error!

			pyArrayCopyConvert(out.e, arrPtr);
		}


		template <typename T>
		static void pyArrayToImageCopy(ImageContainer<T>& out, const PyObject* inPtr)
		{
			// TODO: IMPORTANT! Make ownership semantics for ImageContainer/ImageView
			PyArrayObject* arrPtr = const_cast<PyArrayObject*>(reinterpret_cast<const PyArrayObject*>(inPtr));
			std::size_t ndim = Script::ArrayInfo::getNDims(arrPtr);
			// TODO: Check number of dims?
			// TODO: Do we need to allow overloads for array checks?
			// TODO: Check for contiguous

			ImageDimensions inDims;
			Script::DimType* DIMS = Script::ArrayInfo::getDims(arrPtr);

			for ( int i=0; i < ndim; ++i )
				inDims.dims.e[i] = DIMS[i];

			inDims.chan = (ndim >= 4) ? (DIMS[3]): (1);
			inDims.frame = (ndim >= 5) ? (DIMS[4]): (1);

			out.resize(inDims);
			pyArrayCopyConvert(out.getPtr(), arrPtr);
		}


		// Passthrough conversion (these should already be checked for validity)
		static void convert_impl(const PyArrayObject*& outPtr, const PyObject* inPtr)
		{
			// TODO: Recheck for PyArrayObject here?
			outPtr = reinterpret_cast<const PyArrayObject*>(inPtr);
		}

		// Basic type conversions
		static void convert_impl(uint8_t& out, const PyObject* inPtr)
		{
			unsigned long tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
			{
				// TODO: Throw type-conversion error
			}

			out = static_cast<uint8_t>(tmp);
		}

		static void convert_impl(uint16_t& out, const PyObject* inPtr)
		{
			unsigned long tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
			{
				// TODO: Throw type-conversion error
			}

			out = static_cast<uint16_t>(tmp);
		}

		static void convert_impl(int16_t& out, const PyObject* inPtr)
		{
			long tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
			{
				// TODO: Throw type-conversion error
			}

			out = static_cast<int16_t>(tmp);
		}

		static void convert_impl(uint32_t& out, const PyObject* inPtr)
		{
			unsigned long tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
			{
				// TODO: Throw type-conversion error
			}

			out = static_cast<uint32_t>(tmp);
		}

		static void convert_impl(int32_t& out, const PyObject* inPtr)
		{
			long tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
			{
				// TODO: Throw type-conversion error
			}

			out = static_cast<int32_t>(tmp);
		}

		static void convert_impl(float& out, const PyObject* inPtr)
		{
			double tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
			{
				// TODO: Throw type-conversion error
			}

			out = static_cast<float>(tmp);
		}

		static void convert_impl(double& out, const PyObject* inPtr)
		{
			if ( !pyNumericConvert(&out, const_cast<PyObject*>(inPtr)) )
			{
				// TODO: Throw type-conversion error
			}
		}

		// Vector conversions
		template <typename T>
		static void convert_impl(Vec<T>& out, const PyObject* inPtr)
		{
			if ( PyList_Check(inPtr) )
				pyListToVec(out, inPtr);
			else if ( PyTuple_Check(inPtr) )
				pyTupleToVec(out, inPtr);
			else if ( PyArray_Check(inPtr) )
				pyArrayToVec(out, inPtr);
			else
			{
				// TODO: Throw type error
			}
		}


		// Concrete ImageContainer<T> conversions
		template <typename T>
		static void convert_impl(ImageContainer<T>& out, const PyObject* inPtr)
		{
			if ( PyArray_Check(inPtr) )
			{
				pyArrayToImageCopy(out, inPtr);
			}
			else
			{
				// TODO: Throw type error
			}
		}
	};


	// IO Traits
	template <typename T>
	struct OutParam
	{
		using Type = T;

		template <typename U> using QualT = U;
	};

	template <typename T>
	struct InParam
	{
		using Type = T;

		// TODO: Should really determine this based on whether the data gets copied/converted
		template <typename U>
		using QualT = typename std::conditional<is_deferred<T>::value, force_const_t<U>, U>::type;
	};

	template <typename T>
	struct OptParam
	{
		using Type = T;

		// TODO: Static assert on optional-deferred input (default values can't be type-deferred)
		template <typename U> using QualT = U;
	};





	// TODO: Check for valid io-wrapping
	template <typename T> using is_outparam = std::is_same<T, OutParam<typename T::Type>>;
	template <typename T> using is_inparam = std::is_same<T, InParam<typename T::Type>>;
	template <typename T> using is_optparam = std::is_same<T, OptParam<typename T::Type>>;

	// Generate concrete type from io type
	template <typename T>
	struct concrete_type
	{
		using type = typename T::template QualT<typename T::Type::ConcreteType>;
	};

	template <typename Tuple>
	using concrete_transform = mph::tuple_type_tfm<concrete_type, Tuple>;

	template <typename Tuple>
	using add_ref_transform = mph::tuple_type_tfm<std::add_lvalue_reference, Tuple>;


	template <typename... ArgTypes>
	struct ArgParser
	{
		// TODO: Most of this should go in a base argparser and we can use static
		//   inheritance for Matlab/Python specifics

		// Argument type layout alias (e.g. std::tuple<OutParam<Image<Deferred>>,...>)
		using Layout = std::tuple<ArgTypes...>;
		// Concrete type layouts (e.g. std::tuple<PyObject*,...>)
		using Args = typename concrete_transform<std::tuple<ArgTypes...>>::type;
		using ArgRefs = typename add_ref_transform<Args>::type;

		// Filter sequences for accessing different argument types
		using out_idx_seq = typename mph::filter_tuple_seq<is_outparam, Layout>::type;
		using in_idx_seq = typename mph::filter_tuple_seq<is_inparam, Layout>::type;
		using opt_idx_seq = typename mph::filter_tuple_seq<is_optparam, Layout>::type;

		// Index sequence of input/optional args (in order of required followed by optional)
		using inopt_idx_seq = typename mph::cat_index_sequence<in_idx_seq,opt_idx_seq>::type;

		// IO-type stripped layout subsets (e.g. OutParam<Image<Deferred>> -> Image<Deferred>)
		using OutTypeLayout = mph::tuple_subset_t<out_idx_seq, typename mph::tuple_type_tfm<strip_outer, Layout>::type>;
		using InTypeLayout = mph::tuple_subset_t<in_idx_seq, typename mph::tuple_type_tfm<strip_outer, Layout>::type>;
		using OptTypeLayout = mph::tuple_subset_t<opt_idx_seq, typename mph::tuple_type_tfm<strip_outer, Layout>::type>;

		// Convenience typedefs for concrete output and input argument tuples
		using OutArgs = mph::tuple_subset_t<out_idx_seq, Args>;
		using InArgs = mph::tuple_subset_t<in_idx_seq, Args>;
		using OptArgs = mph::tuple_subset_t<opt_idx_seq, Args>;


		// Sub-sequences of arguments for dealing with type-deferred inputs/outputs
		using inopt_im_idx_seq = typename mph::filter_tuple_subseq<is_image, Layout, inopt_idx_seq>::type;

		using inopt_im_defer_idx_seq = typename mph::filter_tuple_subseq<is_deferred, Layout, inopt_im_idx_seq>::type;
		using out_defer_idx_seq = typename mph::filter_tuple_subseq<is_deferred, Layout, out_idx_seq>::type;

		static constexpr bool has_deferred_image_inputs() noexcept
		{
			return (inopt_im_defer_idx_seq::size() > 0);
		}

		static constexpr bool has_deferred_outputs() noexcept
		{
			return (out_defer_idx_seq::size() > 0);
		}

	public:
		template <typename... T>
		static IdType getInputType(T... ioargs)
		{
			// TODO: Test here for same deferred type? Or in parser?
			auto in_defer_tuple = mph::tuple_subset(inopt_im_defer_idx_seq(), std::tuple<T...>(ioargs...));
			return Script::ArrayInfo::getType(std::get<0>(in_defer_tuple));
		}

		static void parse(ArgRefs ioArgs, PyObject** scriptOut, PyObject* scriptIn)
		{
			// NOTE: Optional argument defaults (in ioArgs) should already be initialized on call to parser
			using ParseInArgs = typename mph::tuple_type_tfm<force_use_type<const_script_in_t>::template tfm, InTypeLayout>::type;
			using ParseOptArgs = typename mph::tuple_type_tfm<force_use_type<const_script_in_t>::template tfm, OptTypeLayout>::type;

			// Local storage for parser output
			ParseInArgs parseInVars;
			ParseOptArgs parseOptVars;

			// Link local vars into reference tuples for parsing and passing around
			auto inParseRefs = mph::tie_tuple(parseInVars);
			auto optParseRefs = mph::tie_tuple(parseOptVars);

			//  Make parse_string and expanded arg tuple for PyParse_Tuple
			const std::string parseStr = make_parse_str<InTypeLayout>() + "|" + make_parse_str<OptTypeLayout>();
			auto parseArgs = std::tuple_cat(expand_parse_args<InTypeLayout>(inParseRefs), expand_parse_args<OptTypeLayout>(optParseRefs));

			//  Call PyParse_Tuple return on error
			if ( !parse_script(scriptIn, parseStr, parseArgs) )
				return;
				// TODO: Throw something instead and handle within main dispatch function

			//  Converters to pass args to actual inputs
			// TODO: Check in-params non-null
			// TODO: Check input image dimension info

			auto inRefs = mph::tuple_subset_ref(in_idx_seq(), ioArgs);
			convert_args(inRefs, inParseRefs);

			auto optRefs = mph::tuple_subset_ref(opt_idx_seq(), ioArgs);
			convert_args(optRefs, optParseRefs);

			// TODO: Support non-deferred outputs (most of this may be deferred)
			// Check for outputs in matlab (e.g. main output)
		}

	private:
		template <typename... TypeLayout, typename... Args, size_t... Is>
		static constexpr auto expand_parse_args_impl(std::tuple<TypeLayout...>, const std::tuple<Args...>& args, mph::index_sequence<Is...>) noexcept
			-> decltype(std::tuple_cat(ParserArg<TypeLayout>::argTuple(std::declval<Args>())...))
		{
			return std::tuple_cat(ParserArg<TypeLayout>::argTuple(std::get<Is>(args))...);
		}

		//////
		// expand_parse_args - Transform PyParser args so they can be passed to PyArg_ParseTuple
		template <typename TypeLayout, typename... Args>
		static constexpr auto expand_parse_args(const std::tuple<Args...>& args) noexcept
			-> decltype(expand_parse_args_impl(std::declval<TypeLayout>(), std::declval<std::tuple<Args...>>(), std::declval<mph::make_index_sequence<sizeof... (Args)>>()))
		{
			return expand_parse_args_impl(TypeLayout(), args, mph::make_index_sequence<sizeof... (Args)>());
		}



		template <typename... Targets, typename... Args, size_t... Is>
		static void convert_args_impl(std::tuple<Targets...>& targets, const std::tuple<Args...>& args, mph::index_sequence<Is...>)
		{
			(void)std::initializer_list<int>
			{
				(ScriptInConvert::convert(std::get<Is>(targets),std::get<Is>(args)), void(), 0)...
			};
		}

		template <typename... Targets, typename... Args>
		static void convert_args(std::tuple<Targets...>& targets, const std::tuple<Args...>& args)
		{
			convert_args_impl(targets, args, mph::make_index_sequence<sizeof... (Args)>());
		}


		// TODO: Change all these to compile-time string classes
		static std::string strcat_initializer(const std::initializer_list<const char*>& strsIn)
		{
			std::string out;
			out.reserve(2*strsIn.size() + 1);
			for ( const auto& it: strsIn )
				out += it;

			return out;
		}

		template <typename... TypeLayout>
		static std::string make_parse_str_impl(std::tuple<TypeLayout...>)
		{
			return strcat_initializer({ ParserArg<TypeLayout>::argString()... });
		}

		template <typename TypeLayout>
		static std::string make_parse_str()
		{
			return make_parse_str_impl(TypeLayout());
		}

		
		template <typename... Args, size_t... Is>
		static bool parse_script_impl(PyObject* scriptIn, const std::string& format, const std::tuple<Args...>& args, mph::index_sequence<Is...>)
		{
			return (PyArg_ParseTuple(scriptIn, format.c_str(), std::get<Is>(args)...) != 0);
		}

		template <typename... Args>
		static bool parse_script(PyObject* scriptIn, const std::string& format, const std::tuple<Args...>& argpack)
		{
			return parse_script_impl(scriptIn, format, argpack, mph::make_index_sequence<sizeof... (Args)>());
		}
	};

};
#pragma once

#include "mph/const_string.h"

#include "ScriptTraits.h"
#include "ScriptTraitTfms.h"

namespace Script
{
	// Derivation of types for script/concrete variable tuples
	template <typename Tuple>
	using script_transform = mph::tuple_type_tfm<iotrait_to_script, Tuple>;

	template <typename Tuple>
	using concrete_transform = mph::tuple_type_tfm<iotrait_to_concrete, Tuple>;

	template <typename OutT, typename InT, typename Tuple>
	using deferred_concrete_transform = mph::tuple_type_tfm<deferred_to_concrete<OutT,InT>::template tfm, Tuple>;

	/////////////
	// ArgConverter - Base structure for script-engine to c++ argument conversions (used by ScriptCommands)
	//   NOTE: These types are generated automatically by GenCommands.h and have fixed argument layouts
	template <typename Derived, typename... Layout>
	struct ArgConverter
	{
	protected:
		using ArgConvertError = typename Script::Converter::ArgConvertError;

		// General argument error exception
		class ArgError: public RuntimeError
		{
			static std::string make_convert_msg(const ArgConvertError& ace)
			{
				return std::string(ace.getArgName()) + ": " + ace.what();
			}

		public:
			ArgError() = delete;

			template <typename... Args>
			ArgError(const char* fmt, Args&&... args)
				: RuntimeError(fmt, std::forward<Args>(args)...)
			{}

			ArgError(const ArgConvertError& ace): ArgError(make_convert_msg(ace).c_str())
			{}
		};

	public:
		// Argument type layout alias (e.g. std::tuple<OutParam<Image<Deferred>>,...>)
		using ArgLayout = std::tuple<Layout...>;

		// Script argument type layout (e.g. std::tuple<const PyArrayObject*,...>
		using ScriptTypes = typename script_transform<std::tuple<Layout...>>::type;
		using ScriptPtrs = typename mph::tuple_ptr_t<ScriptTypes>;

		// Concrete type layouts (e.g. std::tuple<PyObject*,...>)
		using ArgTypes = typename concrete_transform<std::tuple<Layout...>>::type;
		using ArgPtrs = typename mph::tuple_ptr_t<ArgTypes>;

		// Templated concrete deferred type layout
		template <typename OutT, typename InT>
		using ConcreteArgTypes = typename deferred_concrete_transform<OutT,InT,ArgLayout>::type;

		// Compositeable type selectors
		template <typename... Chain> using S_Arg = compose_selector<ArgLayout, true_pred, Chain...>;
		template <typename... Chain> using S_Out = compose_selector<ArgLayout, is_outparam, Chain...>;
		template <typename... Chain> using S_In = compose_selector<ArgLayout, is_inparam, Chain...>;
		template <typename... Chain> using S_Opt = compose_selector<ArgLayout, is_optparam, Chain...>;
		template <typename... Chain> using S_InOpt = S_Arg<S_In<Chain...>, S_Opt<Chain...>>;

		template <typename... Chain> using S_Image = compose_selector<ArgLayout, is_image, Chain...>;

		template <typename... Chain> using S_Defer = compose_selector<ArgLayout, is_deferred, Chain...>;
		template <typename... Chain> using S_Nondef = compose_selector<ArgLayout, not_deferred, Chain...>;

		// Specific composites selector types
		using OutputSel = typename S_Out<>::selector;
		using InOptSel = typename S_InOpt<>::selector;
		using InputSel = typename S_In<>::selector;
		using OptionalSel = typename S_Opt<>::selector;
		using DeferredSel = typename S_Defer<>::selector;
		using DeferredInOptSel = typename S_Defer<S_InOpt<>>::selector;
		using DeferredInImSel = typename S_Defer<S_In<S_Image<>>>::selector;
		using DeferredOutSel = typename S_Defer<S_Out<>>::selector;
		using DeferredOutImSel = typename S_Defer<S_Out<S_Image<>>>::selector;
		using NondeferredSel = typename S_Nondef<>::selector;
		using NondeferOutSel = typename S_Nondef<S_Out<>>::selector;
		using NondeferInOptSel = typename S_Nondef<S_InOpt<>>::selector;

		// Argument layout subsets
		using OutLayout = typename OutputSel::template type<ArgLayout>;
		using InLayout = typename InputSel::template type<ArgLayout>;
		using OptLayout = typename OptionalSel::template type<ArgLayout>;

		// IO-type stripped layout subsets (e.g. OutParam<Image<Deferred>> -> Image<Deferred>)
		using OutTypeLayout = typename mph::tuple_type_tfm<strip_outer, OutLayout>::type;
		using InTypeLayout = typename mph::tuple_type_tfm<strip_outer, InLayout>::type;
		using OptTypeLayout = typename mph::tuple_type_tfm<strip_outer, OptLayout>::type;

		// Optional argument pointers (used for setting defaults)
		using OptPtrs = typename OptionalSel::template type<ArgPtrs>;


	public:
		inline static std::string outargstr()
		{
			return argstr<OutputSel, BracketNone>();
		}

		inline static std::string inoptargstr()
		{
			if ( InputSel::seq::size() > 0 && OptionalSel::seq::size() > 0 )
				return argstr<InputSel, BracketNone>() + "," + argstr<OptionalSel, BracketSquare>();
			if ( OptionalSel::seq::size() > 0 )
				return argstr<OptionalSel, BracketSquare>();
			else
				return argstr<InputSel,BracketNone>();
		}

	private:
		struct BracketNone
		{
			template<std::size_t N>
			static constexpr auto bracketArg(const mph::const_string<N>& str)
				-> const mph::const_string<N>&
			{
				return str;
			}
		};

		struct BracketSquare
		{
			template<std::size_t N>
			static constexpr auto bracketArg(const mph::const_string<N>& str)
				-> mph::const_string<N+2>
			{
				return "[" + str + "]";
			}
		};

		inline std::string comma_delim(const std::string& strA, const std::string& strB)
		{
			if (strA.empty() || strB.empty())
				return strA + strB;
			else
				return strA + "," + strB;
		}

		template <typename ArgSelector, typename BracketType>
		inline static std::string argstr()
		{
			return argstr_impl<BracketType>(typename ArgSelector::seq{});
		}

		template <typename BracketType, std::size_t... Is>
		inline static std::string argstr_impl(mph::index_sequence<Is...>)
		{
			using Seq = mph::index_sequence<Is...>;
			const std::size_t seq_size = Seq::size();

			using CSeq = typename mph::split_sequence<seq_size-1, Seq>::left;
			using Last = typename mph::split_sequence<seq_size-1, Seq>::right;

			return argstr_cat<BracketType>(CSeq(), Last());
		}

		template <typename BracketType>
		inline static std::string argstr_impl(mph::index_sequence<>)
		{
			return "";
		}

		template <typename BrackeType, std::size_t... Is, std::size_t Il>
		inline static std::string argstr_cat(mph::index_sequence<Is...>, mph::index_sequence<Il>)
		{
			// TODO: Bubble constexpr upward if we switch to C++14
			return mph::const_strcat(
				(BrackeType::bracketArg(std::get<Is>(Derived::argNames)) + ",")...,
				BrackeType::bracketArg(std::get<Il>(Derived::argNames)));
		}

	public:
		static constexpr bool has_deferred_image_inputs() noexcept
		{
			return (DeferredInImSel::seq::size() > 0);
		}

		static constexpr bool has_deferred_image_outputs() noexcept
		{
			return (DeferredOutImSel::seq::size() > 0);
		}

		template <typename... Args>
		static IdType getInputType(Args&&... ioargs)
		{
			// TODO: Stop this from erroring if no deferred inputs
			auto in_defer_tuple = DeferredInImSel::select(std::tuple<Args...>(std::forward<Args>(ioargs)...));
			return Script::Array::getType(std::get<0>(in_defer_tuple));
		}

		template <typename... Args>
		static DimInfo getInputDimInfo(const std::tuple<Args...>& argtuple)
		{
			auto in_defer_tuple = DeferredInImSel::select(argtuple);
			return Script::getDimInfo(std::get<0>(in_defer_tuple));
		}

		static void setOptionalDefaults(ArgPtrs argPtrs)
		{
			Derived::setOptional(OptionalSel::select(argPtrs));
		}

		template <typename OutPtrs, typename InPtrs, typename Selector>
		static void convertSelected(OutPtrs outPtrs, InPtrs inPtrs, Selector)
		{
			// Converters to pass script args to actual non-deferred input types
			convert_arg_subset(outPtrs, inPtrs, typename Selector::seq{});
		}

		template <typename ConcretePtrs, typename ScrPtrs>
		static void createOutImRefs(ConcretePtrs cncPtrs, ScrPtrs scrPtrs, const DimInfo& info)
		{
			// TODO: Change image selector (or add new one to make sure this is only for ImageRefs)
			// TODO: Also add static checks since output image owners unsupported
			using BaseTypes = base_data_type_t<ConcretePtrs>;

			mph::tuple_deref(DeferredOutImSel::select(scrPtrs)) = create_arrays<BaseTypes>(info, typename DeferredOutImSel::seq{});
			convert_arg_subset(cncPtrs, scrPtrs, typename DeferredOutImSel::seq{});
		}

	protected:
		//////////////////////////////////
		// Basic type conversions
		// TODO: Fix const-ness issues for Python converters in concrete types
		template <typename T>
		static void convert_impl(T& out, const Script::ObjectType* inPtr)
		{
			out = Converter::toNumeric<T>(const_cast<Script::ObjectType*>(inPtr));
		}

		template <typename T>
		static void convert_impl(Script::ObjectType*& outPtr, const T& in)
		{
			outPtr = Converter::fromNumeric(in);
		}

		static void convert_impl(std::string& out, const Script::ObjectType* inPtr)
		{
			out = Converter::toString(const_cast<Script::ObjectType*>(inPtr));
		}

		template <typename T>
		static void convert_impl(Script::ObjectType*& outPtr, const std::string& in)
		{
			outPtr = Converter::fromString(in);
		}

		// Vector conversions
		template <typename T>
		static void convert_impl(Vec<T>& out, const Script::ObjectType* inPtr)
		{
			out = Converter::toVec<T>(const_cast<Script::ObjectType*>(inPtr));
		}

		template <typename T>
		static void convert_impl(Script::ObjectType*& outPtr, const Vec<T>& in)
		{
			outPtr = Converter::fromVec(in);
		}


		// Concrete ImageOwner<T> conversions
		template <typename T>
		static void convert_impl(ImageOwner<T>& out, const Script::ArrayType* inPtr)
		{
			out = Converter::toImageCopy<T>(const_cast<Script::ArrayType*>(inPtr));
		}


		template <typename T>
		static void convert_impl(ImageView<T>& out, const Script::ArrayType* inPtr)
		{
			out = Converter::toImage<T>(const_cast<Script::ArrayType*>(inPtr));
		}

		template <typename T>
		static void convert_impl(Script::ArrayType*& out, const ImageView<T>& in)
		{
			if ( out == nullptr )
				throw Converter::ImageConvertError("Output image data should already be created");
		}

	protected:
		template <typename T, typename U>
		static void check_convert(T& out, U&& in)
		{
			Derived::convert_impl(out, std::forward<U>(in));
		}

		template <typename T>
		static void check_convert(T& out, const Script::ObjectType* inPtr)
		{
			// NOTE: Arg counts are checked in load() so null/empty is ignored as optional
			if ( inPtr == nullptr || Script::isEmpty(inPtr) )
				return;

			Derived::convert_impl(out, inPtr);
		}

		// Support ArrayType and ObjectType checks (when Script::ArrayType != Script::ObjectType)
		template <typename T, typename U = Script::ArrayType, ENABLE_CHK(!IS_SAME(U,Script::ObjectType))>
		static void check_convert(T& out, const Script::ArrayType* inPtr)
		{
			// NOTE: Arg counts are checked in load() so null/empty is ignored as optional
			if ( inPtr == nullptr || Script::isEmpty(inPtr) )
				return;

			Derived::convert_impl(out, inPtr);
		}

		// Converting input arguments (script types are pointers)
		template <typename OutT, typename InT>
		static void convert_arg(OutT& out, const InT* inPtr, const char* argName)
		{
			try
			{
				Derived::check_convert(out, inPtr);
			}
			catch ( ArgConvertError& ace )
			{
				ace.setArgName(argName);
				throw;
			}
		}

		// Convert output arguments
		template <typename OutT, typename InT>
		static void convert_arg(OutT*& outPtr, const InT& in, const char* argName)
		{
			try
			{
				Derived::check_convert(outPtr, in);
			}
			catch ( ArgConvertError& ace )
			{
				ace.setArgName(argName);
				throw;
			}
		}

		template <typename... Targets, typename... Args, size_t... Is>
		static void convert_arg_subset(std::tuple<Targets*...> targets, std::tuple<Args*...> args, mph::index_sequence<Is...>)
		{
			try
			{
				(void)std::initializer_list<int>
				{
					(convert_arg((*std::get<Is>(targets)), (*std::get<Is>(args)), std::get<Is>(Derived::argNames).c_str()), void(), 0)...
				};
			}
			catch ( ArgConvertError& ace )
			{
				throw ArgError(ace);
			}
		}


		template <typename TargetTuple, size_t... Is>
		static auto create_arrays(const DimInfo& info, mph::index_sequence<Is...>)
			-> decltype(std::make_tuple(Array::create<mph::tuple_select_t<Is, TargetTuple>>(std::declval<const DimInfo&>())...))
		{
			return std::make_tuple(Array::create<mph::tuple_select_t<Is,TargetTuple>>(info)...);
		}
	};
};

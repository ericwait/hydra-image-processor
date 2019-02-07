#pragma once

#include "ScriptTraits.h"
#include "ScriptTraitTfms.h"

namespace Script
{
	// Derivation of types for script/concrete variable tuples
	template <typename Tuple>
	using script_transform = mph::tuple_type_tfm<iotrait_to_script, Tuple>;

	template <typename Tuple>
	using concrete_transform = mph::tuple_type_tfm<iotrait_to_concrete, Tuple>;

	template <typename Tuple>
	using add_ref_transform = mph::tuple_type_tfm<std::add_lvalue_reference, Tuple>;


	template <typename Derived, typename ScriptConverter, typename... Layout>
	struct ArgParser
	{
	protected:
		using ArgConvertError = typename ScriptConverter::ArgConvertError;


	public:
		// Argument type layout alias (e.g. std::tuple<OutParam<Image<Deferred>>,...>)
		using ArgLayout = std::tuple<Layout...>;

		// Script argument type layout (e.g. std::tuple<const PyArrayObject*,...>
		using ScriptTypes = typename script_transform<std::tuple<Layout...>>::type;
		using ScriptRefs = typename add_ref_transform<ScriptTypes>::type;

		// Concrete type layouts (e.g. std::tuple<PyObject*,...>)
		using ArgTypes = typename concrete_transform<std::tuple<Layout...>>::type;
		using ArgRefs = typename add_ref_transform<ArgTypes>::type;

		// Filter sequences for accessing different argument types
		using out_idx_seq = typename mph::filter_tuple_seq<is_outparam, ArgLayout>::type;
		using in_idx_seq = typename mph::filter_tuple_seq<is_inparam, ArgLayout>::type;
		using opt_idx_seq = typename mph::filter_tuple_seq<is_optparam, ArgLayout>::type;

		// Index sequence of input/optional args (in order of required followed by optional)
		using inopt_idx_seq = typename mph::cat_index_sequence<in_idx_seq, opt_idx_seq>::type;

		// IO-type stripped layout subsets (e.g. OutParam<Image<Deferred>> -> Image<Deferred>)
		using OutTypeLayout = mph::tuple_subset_t<out_idx_seq, typename mph::tuple_type_tfm<strip_outer, ArgLayout>::type>;
		using InTypeLayout = mph::tuple_subset_t<in_idx_seq, typename mph::tuple_type_tfm<strip_outer, ArgLayout>::type>;
		using OptTypeLayout = mph::tuple_subset_t<opt_idx_seq, typename mph::tuple_type_tfm<strip_outer, ArgLayout>::type>;

		// Convenience typedefs for concrete output and input argument tuples
		using OutArgs = mph::tuple_subset_t<out_idx_seq, ArgTypes>;
		using InArgs = mph::tuple_subset_t<in_idx_seq, ArgTypes>;
		using OptArgs = mph::tuple_subset_t<opt_idx_seq, ArgTypes>;


		// Sub-sequences of arguments for dealing with type-deferred inputs/outputs
		using in_im_idx_seq = typename mph::filter_tuple_subseq<is_image, ArgLayout, in_idx_seq>::type;

		using in_im_defer_idx_seq = typename mph::filter_tuple_subseq<is_deferred, ArgLayout, in_im_idx_seq>::type;
		using out_defer_idx_seq = typename mph::filter_tuple_subseq<is_deferred, ArgLayout, out_idx_seq>::type;


	public:
		static constexpr bool has_deferred_image_inputs() noexcept
		{
			return (in_im_defer_idx_seq::size() > 0);
		}

		static constexpr bool has_deferred_outputs() noexcept
		{
			return (out_defer_idx_seq::size() > 0);
		}

		template<typename... Types>
		static constexpr auto selectOptional(std::tuple<Types...>& args)
			-> mph::tuple_subset_t<opt_idx_seq, std::tuple<typename std::add_lvalue_reference<Types>::type...>>
		{
			return mph::tuple_subset(opt_idx_seq(), args);
		}

		template<typename... Types>
		static constexpr auto selectInputs(std::tuple<Types...>& argsRefs)
			-> mph::tuple_subset_t<in_idx_seq, std::tuple<typename std::add_lvalue_reference<Types>::type...>>
		{
			return mph::tuple_subset(in_idx_seq(), argsRefs);
		}

		template<typename... Types>
		static constexpr auto selectOutputs(std::tuple<Types...>& argsRefs)
			-> mph::tuple_subset_t<out_idx_seq, std::tuple<typename std::add_lvalue_reference<Types>::type...>>
		{
			return mph::tuple_subset(out_idx_seq(), argsRefs);
		}

		template <typename... T>
		static IdType getInputType(const T&... ioargs)
		{
			auto in_defer_tuple = mph::tuple_subset(in_im_defer_idx_seq(), std::tuple<const T&...>(ioargs...));
			return Script::ArrayInfo::getType(std::get<0>(in_defer_tuple));
		}


		static void convertInputs(ArgRefs ioArgs, ScriptRefs scriptRefs)
		{
			// TODO: Potentially pre-check for conversion compatibility
			//  Converters to pass script args to actual non-deferred input types
			convert_arg_subset(ioArgs, scriptRefs, inopt_idx_seq());
		}

		template <typename InT>
		static void convertDeferredInputs(ArgRefs concreteArgs, ArgRefs ioArgs);



	public:
		// General argument error exception
		class ArgError: public std::runtime_error
		{
			static std::string make_convert_msg(const ArgConvertError& ace)
			{
				return std::string(Derived::argName(ace.getArgIndex())) + ": " + ace.what();
			}

		public:
			ArgError(const char* msg): std::runtime_error(msg)
			{}

			ArgError(const ArgConvertError& ace): std::runtime_error(make_convert_msg(ace))
			{}

		private:
			ArgError() = delete;
		};

	protected:
		template <typename... Targets, typename... Args, size_t... Is>
		static void convert_arg_subset(std::tuple<Targets...>& targets, const std::tuple<Args...>& args, mph::index_sequence<Is...>)
		{
			try
			{
				(void)std::initializer_list<int>
				{
					(ScriptConverter::convert(std::get<Is>(targets), std::get<Is>(args), Is), void(), 0)...
				};
			}
			catch ( ArgConvertError& ace )
			{
				throw ArgError(ace);
			}
		}

		//template <typename... Targets, typename... Args>
		//static void convert_args(std::tuple<Targets...>& targets, const std::tuple<Args...>& args)
		//{
		//	convert_args_subset(targets, args, mph::make_index_sequence<sizeof... (Args)>());
		//}
	};
};

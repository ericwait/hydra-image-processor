#pragma once

#include "mph/qualifier_helpers.h"
#include "mph/tuple_helpers.h"
#include "ScriptTraits.h"


namespace Script
{
	/////////////////////////
	// strip_outer -
	//   Remove one layer of nested types (used to remove In/Out/OptParam qualifier types)
	/////////////////////////
	template <typename T>
	struct strip_outer {};

	template <template<typename> class T, typename U>
	struct strip_outer<T<U>>
	{
		using type = U;
	};


	template <typename RawType, typename Type>
	struct base_type_impl
	{
		using type = Type;
	};

	template <template <typename> class Trait, typename BaseType, typename Type>
	struct base_type_impl<Trait<BaseType>, Type>
	{
		using type = typename base_type_impl<mph::raw_type_t<BaseType>,BaseType>::type;
	};


	/////////////////////////
	// base_type -
	//   Produce underlying contained type (e.g. ImageView<int>& -> int)
	/////////////////////////
	template <typename Type>
	struct base_type
	{
		using type = typename base_type_impl<mph::raw_type_t<Type>,Type>::type;
	};


	template <typename Tuple>
	using base_data_type_t = typename mph::tuple_type_tfm<base_type, Tuple>::type;


	//////////////
	// Data type conversions (compile-time) for use with script trait types

	/////////////////////////
	// dtrait_to_script -
	//   Transforms from data-trait type to script base-type
	//   (e.g. Image<int> -> Script::ArrayType*)
	/////////////////////////
	template <typename Traits>
	struct dtrait_to_script {};

	template <template <typename> class DataTrait, typename BaseType>
	struct dtrait_to_script<DataTrait<BaseType>>
	{
		using type = Script::ObjectType*;
	};

	template <typename BaseType>
	struct dtrait_to_script<Image<BaseType>>
	{
		using type = Script::ArrayType*;
	};

	template <typename BaseType>
	struct dtrait_to_script<ImageRef<BaseType>>
	{
		using type = Script::ArrayType*;
	};

	/////////////////////////
	// dtrait_to_scriptout -
	//   Transforms from data-trait type to script base-type (output only)
	//   (e.g. Image<int> -> Script::GuardOutObjectPtr)
	/////////////////////////
	template <typename Traits>
	struct dtrait_to_scriptout {};

	template <template <typename> class DataTrait, typename BaseType>
	struct dtrait_to_scriptout<DataTrait<BaseType>>
	{
		static_assert(!std::is_same<DataTrait<BaseType>,Image<BaseType>>::value, "HIP_COMPILE: Only ImageRefs (not Images) are allowed as script outputs");
		using type = typename dtrait_to_script<DataTrait<BaseType>>::type;
	};

	template <typename BaseType>
	struct dtrait_to_scriptout<StructTrait<BaseType>>
	{
		using type = Script::GuardOutObjectPtr;
	};

	template <typename BaseType>
	struct dtrait_to_scriptout<ImageRef<BaseType>>
	{
		using type = Script::GuardOutArrayPtr;
	};

	/////////////////////////
	// iotrait_to_script -
	//   Transforms from full io-trait type to base script type.
	//   (e.g. InParam<Image<Deferred>> -> const Script::ArrayType*)
	/////////////////////////
	template <typename Traits>
	struct iotrait_to_script {};

	template <template <typename> class IOTrait, typename DataTraits>
	struct iotrait_to_script<IOTrait<DataTraits>>
	{
		using type = mph::force_const_t<typename dtrait_to_script<DataTraits>::type>;
	};

	template <typename DataTraits>
	struct iotrait_to_script<OutParam<DataTraits>>
	{
		using type = typename dtrait_to_scriptout<DataTraits>::type;
	};

	/////////////////////////
	// dtrait_to_concrete -
	//   Transforms from data-trait type to concrete base-type
	//   (e.g. ImageRef<int> -> ImageView<int>)
	/////////////////////////
	template <typename Traits>
	struct dtrait_to_concrete {};

	template <typename BaseType>
	struct dtrait_to_concrete<Scalar<BaseType>>
	{
		using type = BaseType;
	};

	template <>
	struct dtrait_to_concrete<Scalar<DeferredType>>
	{
		using type = Script::ObjectType*;
	};

	template <typename BaseType>
	struct dtrait_to_concrete<Vector<BaseType>>
	{
		using type = Vec<BaseType>;
	};

	template <>
	struct dtrait_to_concrete<Vector<DeferredType>>
	{
		using type = Script::ObjectType*;
	};

	template <typename BaseType>
	struct dtrait_to_concrete<Image<BaseType>>
	{
		using type = ImageOwner<BaseType>;
	};

	template <>
	struct dtrait_to_concrete<Image<DeferredType>>
	{
		using type = Script::ArrayType*;
	};

	template <typename BaseType>
	struct dtrait_to_concrete<ImageRef<BaseType>>
	{
		using type = ImageView<BaseType>;
	};

	template <>
	struct dtrait_to_concrete<ImageRef<DeferredType>>
	{
		using type = Script::ArrayType*;
	};

	/////////////////////////
	// iotrait_to_concrete -
	//   Transforms from full io-trait types to concrete base types
	//   (e.g. InParam<Image<int>> -> ImageContainer<int>)
	//   NOTE: This leaves deferred types the same as iotrait_to_script
	/////////////////////////
	template <typename Traits>
	struct iotrait_to_concrete {};

	template <template <typename> class IOTrait, template <typename> class DataTrait, typename BaseType>
	struct iotrait_to_concrete<IOTrait<DataTrait<BaseType>>>
	{
		// Use script types for deferred io traits
		using type = typename dtrait_to_concrete<DataTrait<BaseType>>::type;
	};

	template <template <typename> class IOTrait, template <typename> class DataTrait>
	struct iotrait_to_concrete<IOTrait<DataTrait<DeferredType>>>
	{
		using type = typename iotrait_to_script<IOTrait<DataTrait<DeferredType>>>::type;
	};

	/////////////////////////
	// deferred_dtrait_to_concrete -
	//   Transforms all data-trait types to concrete types
	//   NOTE: The deferred types are all converted using the extra T parameter
	/////////////////////////
	template <typename Traits, typename T>
	struct deferred_dtrait_to_concrete {};

	template <template <typename> class DataTrait, typename BaseType, typename T>
	struct deferred_dtrait_to_concrete<DataTrait<BaseType>, T>
	{
		using type = typename dtrait_to_concrete<DataTrait<BaseType>>::type;
	};

	template <template <typename> class DataTrait, typename T>
	struct deferred_dtrait_to_concrete<DataTrait<DeferredType>, T>
	{
		using type = typename dtrait_to_concrete<DataTrait<T>>::type;
	};


	/////////////////////////
	// deferred_iotrait_to_concrete_impl -
	//   Transforms from full io-trait types to concrete base types (including deferred types)
	//   NOTE: The deferred types are all converted using the extra OutT/InT parameters
	/////////////////////////
	template <typename Traits, typename OutT, typename InT>
	struct deferred_iotrait_to_concrete_impl {};

	template <template <typename> class IOTrait, typename DataTraits, typename OutT, typename InT>
	struct deferred_iotrait_to_concrete_impl<IOTrait<DataTraits>, OutT, InT>
	{
		using type = typename deferred_dtrait_to_concrete<DataTraits, InT>::type;
	};

	template <typename DataTraits, typename OutT, typename InT>
	struct deferred_iotrait_to_concrete_impl<OutParam<DataTraits>, OutT, InT>
	{
		using type = typename deferred_dtrait_to_concrete<DataTraits, OutT>::type;
	};

	/////////////////////////
	// deferred_to_concrete -
	//   Transforms all io-trait types to concrete types
	//   NOTE: The deferred types are all converted using the extra OutT,InT parameters
	/////////////////////////
	template <typename OutT, typename InT>
	struct deferred_to_concrete
	{
		template <typename Traits>
		struct tfm
		{
			using type = typename deferred_iotrait_to_concrete_impl<Traits, OutT, InT>::type;
		};
	};


	/////////////////////////
	// arg_selector -
	//   Helper class used to select argument subsets (built by compose_selector)
	/////////////////////////
	template <typename Seq>
	struct arg_selector {};

	template <std::size_t... Is>
	struct arg_selector<mph::index_sequence<Is...>>
	{
		using seq = typename mph::index_sequence<Is...>;

		template <typename Tuple>
		using type = typename mph::tuple_subset_t<seq, Tuple>;

		template <typename Tuple>
		static constexpr auto select(Tuple&& tuple)
			-> decltype(mph::tuple_subset(seq{}, std::declval<Tuple>()))
		{
			return mph::tuple_subset(seq{}, std::forward<Tuple>(tuple));
		}
	};


	/////////////////////////
	// compose_selector -
	//   Composable predicate chains used to build arg_selector<> types
	/////////////////////////
	template <typename Layout, template <typename> class Pred, typename... Chain>
	struct compose_selector {};

	template <typename... Types, template <typename> class Pred, typename... Chain>
	struct compose_selector<std::tuple<Types...>, Pred, Chain...>
	{
		using seq = typename mph::cat_index_sequence<typename mph::filter_tuple_subseq<Pred, std::tuple<Types...>, typename Chain::seq>::type...>::type;
		using selector = arg_selector<seq>;
	};

	template <typename... Types, template <typename> class Pred>
	struct compose_selector<std::tuple<Types...>, Pred>
	{
		using seq = typename mph::filter_tuple_seq<Pred, std::tuple<Types...>>::type;
		using selector = arg_selector<seq>;
	};

};

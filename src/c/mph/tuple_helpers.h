#pragma once

#include "integer_sequence.h"

namespace mph
{
	// Internals for tuple-predicate filtering
	namespace internal
	{
		template <bool> struct conditional_seq { template <std::size_t I> using type = index_sequence<I>; };
		template <> struct conditional_seq<false> { template <std::size_t I> using type = index_sequence<>; };

		template <template<typename> class Pred, typename Tuple, typename Seq>
		struct filter_tuple_seq_impl {};

		template <template<typename> class Pred, typename Head, typename... Tail, std::size_t IH, std::size_t... IT>
		struct filter_tuple_seq_impl<Pred, std::tuple<Head, Tail...>, index_sequence<IH, IT...>>
		{
			using type = typename mph::cat_integer_sequence<std::size_t,
				typename conditional_seq<Pred<Head>::value>::template type<IH>,
				typename filter_tuple_seq_impl<Pred, std::tuple<Tail...>, index_sequence<IT...>>::type>::type;
		};

		template <template<typename> class Pred>
		struct filter_tuple_seq_impl<Pred, std::tuple<>, index_sequence<>>
		{
			using type = index_sequence<>;
		};


		// Used for index_seqence filtering
		template <size_t I, typename Head, typename... Tail>
		struct select_type_impl : public select_type_impl<I-1, Tail...>
		{};

		template <typename Head, typename... Tail>
		struct select_type_impl<0, Head, Tail...>
		{
			using type = Head;
		};


		// Get types of a tuple filtered on a index_sequence
		template <typename Seq, typename Tuple>
		struct select_tuple_types{};

		template <std::size_t... Is, typename... Types>
		struct select_tuple_types<index_sequence<Is...>, std::tuple<Types...>>
		{
			using type = std::tuple<typename select_type_impl<Is, Types...>::type...>;
		};
	};

	/////////////////////////
	// filter_tuple_seq -
	//   Generates an index sequence that can be used to select tuple subsets
	//   based on a predicate (true/false) transform applied to each tuple element type
	/////////////////////////
	template <template<typename> class Pred, typename Tuple>
	struct filter_tuple_seq {};

	template <template<typename> class Pred, typename... Types>
	struct filter_tuple_seq<Pred, std::tuple<Types...>>
	{
		using type = typename internal::filter_tuple_seq_impl<Pred, std::tuple<Types...>, make_index_sequence<sizeof... (Types)>>::type;
	};

	/////////////////////////
	// filter_tuple_subseq -
	//   Generates an index sequence that can be used to select tuple subsets
	//   based on a predicate (true/false) transform applied to each tuple element type
	//
	//   This version takes an input sub-seqeuence which is applied to tuple as a pre-filter
	//   NOTE: Used for compositing filters (equivalent to and-ing predicates)
	/////////////////////////
	template <template<typename> class Pred, typename Tuple, typename Subseq>
	struct filter_tuple_subseq {};

	template <template<typename> class Pred, typename... Types, size_t... Is>
	struct filter_tuple_subseq<Pred, std::tuple<Types...>, index_sequence<Is...>>
	{
		using filtered_tuple = typename internal::select_tuple_types<index_sequence<Is...>, std::tuple<Types...>>::type;
		using type = typename internal::filter_tuple_seq_impl<Pred, filtered_tuple, index_sequence<Is...>>::type;
	};


	// Internal implementation for tie_tuple
	namespace internal
	{
		// TODO: Play with std::forward for this?
		template <std::size_t... Is, typename... Types>
		constexpr std::tuple<Types&...> tie_tuple_impl(index_sequence<Is...>, std::tuple<Types...>& tuple)
		{
			return std::tuple<Types&...>(std::get<Is>(tuple)...);
		}
	};

	/////////////////////////
	// tie_tuple -
	//   Takes a tuple instance and outputs a tuple of references to the tied variables
	//   NOTE: the use of std::tie means this is really only useful if you have primary
	//   data stored in a tuple
	/////////////////////////
	template <typename... Types>
	constexpr std::tuple<Types&...> tie_tuple(std::tuple<Types...>& tuple)
	{
		return internal::tie_tuple_impl(make_index_sequence<sizeof...(Types)>(), tuple);
	}


	/////////////////////////
	// tuple_subset_t -
	//   Returns the subset of tuple-types selected from the index sequence
	/////////////////////////
	template <typename Seq, typename Tuple>
	using tuple_subset_t = typename internal::select_tuple_types<Seq, Tuple>::type;


	/////////////////////////
	// tuple_subset -
	//   Returns a new tuple that is the subset of elements (value-copied) listed in the
	//   index sequence
	/////////////////////////
	template <std::size_t... Is, typename... Types>
	constexpr auto tuple_subset(index_sequence<Is...>, const std::tuple<Types...>& vars)
		-> tuple_subset_t<index_sequence<Is...>, std::tuple<Types...>>
	{
		return tuple_subset_t<index_sequence<Is...>, std::tuple<Types...>>(std::get<Is>(vars)...);
	}

	/////////////////////////
	// tuple_subset_ref -
	//   Returns a new tuple that is the subset of elements (references) listed in the
	//   index sequence
	//   NOTE: This SHOULD fail at compile-time if passed a make_tuple (doesn't on msvc)
	/////////////////////////
	template <std::size_t... Is, typename... Types>
	constexpr auto tuple_subset_ref(index_sequence<Is...>, std::tuple<Types...>& vars)
		-> tuple_subset_t<index_sequence<Is...>, std::tuple<Types&...>>
	{
		return tuple_subset_t<index_sequence<Is...>, std::tuple<Types&...>>(std::get<Is>(vars)...);
	}


	/////////////////////////
	// tuple_type_tfm -
	//   Applies a type-transform structure (Tfm<E>::type) to each element-type E of a tuple
	//   returns a tuple-type of transformed types
	/////////////////////////
	template <template<typename> class Tfm, typename Tuple>
	struct tuple_type_tfm {};

	template <template<typename> class Tfm, typename... Args>
	struct tuple_type_tfm<Tfm, std::tuple<Args...>>
	{
		using type = std::tuple<typename Tfm<Args>::type...>;
	};


	template <template<typename> class TfmH, template<typename> class... TfmTail>
	struct compose_type_tfm
	{
		template <typename Tuple>
		struct tfm
		{
			using type = typename TfmH<
				typename compose_type_tfm<TfmTail...>::template tfm<Tuple>::type
				>::type;
		};
	};

	template <template<typename> class TfmH>
	struct compose_type_tfm<TfmH>
	{
		template <typename Tuple>
		struct tfm
		{
			using type = typename TfmH<Tuple>::type;
		};
	};


	namespace internal
	{
		template <typename Tuple>
		using composed_deref = compose_type_tfm<std::add_lvalue_reference, std::remove_pointer, std::remove_reference>::template tfm<Tuple>;
	};


	template <typename Tuple>
	using tuple_ptr_t = typename tuple_type_tfm<std::add_pointer, Tuple>::type;

	template <typename Tuple>
	using tuple_deref_t = typename tuple_type_tfm<internal::composed_deref, Tuple>::type;

	namespace internal
	{
		template <typename... Args, std::size_t... Is>
		constexpr tuple_ptr_t<std::tuple<Args...>> tuple_addr_of_impl(index_sequence<Is...>, std::tuple<Args...>& vars)
		{
			return std::make_tuple((&std::get<Is>(vars))...);
		}

		template <typename... Args, std::size_t...Is>
		constexpr tuple_deref_t<std::tuple<Args...>> tuple_deref_impl(index_sequence<Is...>, std::tuple<Args...>& vars)
		{
			return std::tie((*std::get<Is>(vars))...);
		}
	};

	/////////////////////////
	// tuple_addr_of -
	//   Return a tuple of pointers to input tuple elements (e.g. &var for each element)
	/////////////////////////
	template <typename... Args>
	constexpr tuple_ptr_t<std::tuple<Args...>> tuple_addr_of(std::tuple<Args...>& tuple)
	{
		return internal::tuple_addr_of_impl(make_index_sequence<sizeof... (Args)>(), tuple);
	}


	template <typename... Args>
	constexpr tuple_deref_t<std::tuple<Args...>> tuple_deref(std::tuple<Args...> tuple)
	{
		return internal::tuple_deref_impl(make_index_sequence<sizeof... (Args)>(), tuple);
	}


	// TODO: Move these to a different mph header
	template <bool B>
	using bool_constant = std::integral_constant<bool, B>;

	template<class B>
	struct negation: bool_constant<!bool(B::value)> { };

	template <class T> typename std::add_rvalue_reference<T>::type declval();
};

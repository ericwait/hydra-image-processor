#pragma once

#include <cstddef>

namespace mph
{
	// TODO: Just use c++14 in compilation?

	/////////////////////////
	// integer_sequence - (C++14)
	//   Represents a sequence of integers of type T as integer_sequence<T, I1,I2,...,In>
	//   NOTE: Though they can only be created in order (using make_integer_sequence) 
	//     They can be filtered and concatenated to produce sequences of arbitrary order.
	/////////////////////////
	template <typename T, T... N>
	struct integer_sequence
	{
		static_assert(std::is_integral<T>::value, "Integer sequence type must be integral");

		using value_type = T;
		inline static constexpr std::size_t size() noexcept
		{
			return (sizeof... (N));
		}
	};

	/////////////////////////
	// index_sequence - (C++14)
	//   Convenience type around integer_sequence<size_t,...>
	/////////////////////////
	template <std::size_t... I>
	using index_sequence = integer_sequence<std::size_t, I...>;

	/////////////////////////
	// cat_integer_sequence -
	//   Concatenate integer sequences together
	//   NOTE: Also used internally for generating sequences
	/////////////////////////
	template <typename T, typename... Seq>
	struct cat_integer_sequence {};

	template <typename T, T... Ia, T... Ib, typename... Tail>
	struct cat_integer_sequence<T, integer_sequence<T, Ia...>, integer_sequence<T, Ib...>, Tail...>
	{
		using type = typename cat_integer_sequence<T, integer_sequence<T, Ia..., Ib...>, Tail...>::type;
	};

	// End case for cat with a single sequence
	template <typename T, T... Is>
	struct cat_integer_sequence<T, integer_sequence<T, Is...>>
	{
		using type = integer_sequence<T, Is...>;
	};

	// Return empty sequence if called with no args
	template <typename T>
	struct cat_integer_sequence<T>
	{
		using type = integer_sequence<T>;
	};

	template <typename... Seq>
	using cat_index_sequence = cat_integer_sequence<std::size_t, Seq...>;


	// Internals for making integer sequences
	namespace internal
	{
		// Can't use dependent types in template to generate sequences so use std::size_t and convert
		template <typename T, typename I>
		struct convert_integer_sequence {};

		template <typename T, typename U, U... Is>
		struct convert_integer_sequence<T, integer_sequence<U, Is...>>
		{
			using type = integer_sequence<T, Is...>;
		};


		// Generate index_sequence<0,...,N>
		template <std::size_t N>
		struct gen_sequence
		{
			using type = typename cat_index_sequence<typename gen_sequence<N-1>::type, index_sequence<N>>::type;
		};

		template <>
		struct gen_sequence<0>
		{
			using type = index_sequence<0>;
		};

		// Guard to make sure zero sequences don't get passed in
		template <std::size_t N>
		struct gen_seq_guard
		{
			using type = typename gen_sequence<N-1>::type;
		};

		template <>
		struct gen_seq_guard<0>
		{
			using type = integer_sequence<std::size_t>;
		};


		template <typename T, T N>
		struct make_int_seq_impl
		{
			using type = typename convert_integer_sequence<T, typename gen_seq_guard<N>::type>::type;
		};


		// Useful helper type (conditionally produces a sequence element or empty sequence>
		template <bool> struct conditional_seq { template <typename T, T I> using type = integer_sequence<T,I>; };
		template <> struct conditional_seq<false> { template <typename T, T I> using type = integer_sequence<T>; };


		// Select a sequence element
		template <std::size_t N, typename Seq>
		struct nth_seq_elem{};

		template <std::size_t N, typename T, T I, T... IT>
		struct nth_seq_elem<N, integer_sequence<T, I, IT...>>
			: nth_seq_elem<N-1, integer_sequence<T, IT...>> {};

		template <typename T, T I, T... IT>
		struct nth_seq_elem<0, integer_sequence<T, I, IT...>>
		{
			static constexpr const T elem = I;
		};

		template <typename Seq, typename Subseq>
		struct select_subseq_impl {};

		template <typename T, T... Is, std::size_t... Isub>
		struct select_subseq_impl<integer_sequence<T,Is...>, index_sequence<Isub...>>
		{
			using type = integer_sequence<T, nth_seq_elem<Isub, integer_sequence<T,Is...>>::elem...>;
		};

		template <std::size_t N, typename Seq, typename CountSeq>
		struct split_sequence_impl {};

		template <std::size_t N, typename T, T... Is, std::size_t... Cs>
		struct split_sequence_impl<N, integer_sequence<T,Is...>, index_sequence<Cs...>>
		{
			using left = typename cat_integer_sequence<T, typename conditional_seq<Cs < N >::template type<T,Is>...>::type;
			using right = typename cat_integer_sequence<T, typename conditional_seq<Cs >= N>::template type<T,Is>...>::type;
		};
	};

	/////////////////////////
	// make_integer_sequence - (C++14)
	//   Make an integer sequence -> integer_sequence<T, 0,...,N-1>
	/////////////////////////
	template <typename T, T N>
	using make_integer_sequence = typename internal::make_int_seq_impl<T, N>::type;


	/////////////////////////
	// make_index_sequence -
	//   Make index sequence from 0 to N-1 -> index_sequence<0,...,N-1>
	/////////////////////////
	template <std::size_t N>
	using make_index_sequence = make_integer_sequence<std::size_t, N>;


	/////////////////////////
	// split_sequence -
	//   Splits an integer sequence into left/right sequences at the Nth element
	//   (e.g. split_sequence<2, index_sequence<0,1,2,3,4>>
	//           -> left=index_sequence<0,1>, right=index_sequence<2,3,4>
	/////////////////////////
	template <std::size_t N, typename Seq>
	struct split_sequence : internal::split_sequence_impl<N, Seq, make_index_sequence<Seq::size()>>
	{};

};

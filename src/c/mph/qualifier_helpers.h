#pragma once

#include <type_traits>

namespace mph
{
	namespace internal
	{
		// This helper class provides the ability to strip away and rebuild pointer/references
		//   For example traverse_type<int**>::type -> int*
		// NOTE: This is used for applying qualifier type transformations
		template <typename T>
		struct traverse_type
		{
			template <typename U> using back = U;
			using type = std::nullptr_t;
		};

		// Traversal guard (will fail to compile if user structure doesn't specialize nullptr_t)
		template <>
		struct traverse_type<std::nullptr_t> {};

		template <typename T>
		struct traverse_type<T&>
		{
			template <typename U> using back = typename std::add_lvalue_reference<U>::type;
			using type = T;
		};

		template <typename T>
		struct traverse_type<T&&>
		{
			template <typename U> using back = typename std::add_rvalue_reference<U>::type;
			using type = T;
		};

		template <typename T>
		struct traverse_type<T* const>
		{
			template <typename U> using back = typename std::add_pointer<typename std::add_const<U>::type>::type;
			using type = T;
		};

		template <typename T>
		struct traverse_type<T* volatile>
		{
			template <typename U> using back = typename std::add_pointer<typename std::add_volatile<U>::type>::type;
			using type = T;
		};

		template <typename T>
		struct traverse_type<T* const volatile>
		{
			template <typename U> using back = typename std::add_pointer<typename std::add_cv<U>::type>::type;
			using type = T;
		};

		template <typename T>
		struct traverse_type<T*>
		{
			template <typename U> using back = typename std::add_pointer<U>::type;
			using type = T;
		};


		template <typename T, typename U>
		struct raw_type_impl
		{
			using type = typename raw_type_impl<U, typename traverse_type<U>::type>::type;
		};

		template <typename T>
		struct raw_type_impl<T,std::nullptr_t>
		{
			using type = T;
		};



		template <template <typename> class Tfm, typename T, typename U>
		struct data_type_tfm_impl
		{
			using type = typename traverse_type<U>::template back<
				typename data_type_tfm_impl<Tfm, U, typename traverse_type<U>::type>::type>;
		};

		template <template <typename> class Tfm, typename T>
		struct data_type_tfm_impl<Tfm, T, std::nullptr_t>
		{
			using type = typename Tfm<T>::type;
		};

		template <template <typename> class Tfm, typename T, typename U>
		struct full_type_tfm_impl
		{
			using type = typename Tfm<
				typename traverse_type<U>::template back<
				typename full_type_tfm_impl<Tfm, U, typename traverse_type<U>::type>::type>
			>::type;
		};

		template <template <typename> class Tfm, typename T>
		struct full_type_tfm_impl<Tfm, T, std::nullptr_t>
		{
			using type = T;
		};
	};

	/////////////////////////
	// raw_type -
	//   Remove all qualifiers and pointer/references to get underlying type (const int**& -> int)
	/////////////////////////
	template <typename T>
	using raw_type = internal::raw_type_impl<T,T>;

	template <typename T>
	using raw_type_t = typename raw_type<T>::type;


	/////////////////////////
	// data_type_tfm_t -
	//   Apply a transform to the underlying data type (e.g. data_type_tfm_t<add_const, int**> -> (const int)** )
	/////////////////////////
	template <template <typename> class Tfm, typename T>
	using data_type_tfm_t = typename internal::data_type_tfm_impl<Tfm, T, T>::type;

	/////////////////////////
	// full_type_tfm_t -
	//   Apply a transform every layer of a type (e.g. full_type_tfm_t<add_const, int**> -> const int * const * const )
	/////////////////////////
	template <template <typename> class Tfm, typename T>
	using full_type_tfm_t = typename internal::full_type_tfm_impl<Tfm, T, T>::type;

	/////////////////////////
	// void_t - C++14 compatibility (used with SFINAE for identifying "valid" type expressions)
	/////////////////////////
	template <typename... Ts> using void_t = void;
};

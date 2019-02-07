#pragma once

#include "mph/tuple_helpers.h"
#include "mph/qualifier_helpers.h"

namespace Script
{
	// Deferred-type is used to indicate types identified dynamically
	struct DeferredType {};

	// Supported script input types (scalars, 3-vectors, 5-D images)
	template <typename T> struct Scalar {};
	template <typename T> struct Vector {};
	template <typename T> struct Image {};
	template <typename T> struct ImageRef {};


	// IO Traits (Parameter type, output, input, optional input)
	template <typename T>
	struct OutParam
	{};

	template <typename T>
	struct InParam
	{};

	template <typename T>
	struct OptParam
	{
		// TODO: Static assert on invalid- io type combinations (e.g. default values can't be type-deferred)
	};

	// NOTE: These are general predicates for interrogating nested types
	/////////////////////////
	// has_underlying_type -
	//   Predicate: true if underlying type is U (e.g. OutParam<ImageRef<V>>)
	/////////////////////////
	template <typename T, typename U>
	struct has_underlying_type
	{
		static constexpr bool value = std::is_same<T,U>::value;
	};

	template <template<typename> class T, typename V, typename U>
	struct has_underlying_type<T<V>, U>
	{
		static constexpr bool value = has_underlying_type<V,U>::value;
	};


	/////////////////////////
	// has_trait -
	//   Predicate: true if contains a type-trait (Trait) in nested type set (e.g. A<Trait<B<...>>>)
	/////////////////////////
	template <typename T, template<typename> class Trait>
	struct has_trait : std::false_type
	{};

	template <template<typename> class T, typename V, template<typename> class  Trait>
	struct has_trait<T<V>, Trait>
	{
		static constexpr bool value = (std::is_same<T<V>,Trait<V>>::value || has_trait<V,Trait>::value);
	};


	/////////////////////////
	// is_deferred -
	//   Predicate: true if underlying type is deferred (e.g. OutParam<ImageRef<DeferredType>>)
	/////////////////////////
	template <typename T>
	struct is_deferred
	{
		static constexpr bool value = has_underlying_type<T, DeferredType>::value;
	};


	/////////////////////////
	// is_image -
	//   Predicate: true if contains an image/imageref (e.g. InParam<ImageRef<DeferredType>>)
	/////////////////////////
	template <typename T>
	struct is_image
	{
		static constexpr bool value = (has_trait<T, ImageRef>::value || has_trait<T, Image>::value);
	};


	/////////////////////////
	// is_outparam -
	//   Predicate: true if is output parameter (e.g. OutParam<...>)
	/////////////////////////
	template <typename T>
	struct is_outparam
	{
		static constexpr bool value = has_trait<T, OutParam>::value;
	};

	/////////////////////////
	// is_inparam -
	//   Predicate: true if is input parameter (e.g. InParam<...>)
	/////////////////////////
	template <typename T>
	struct is_inparam
	{
		static constexpr bool value = has_trait<T, InParam>::value;
	};

	/////////////////////////
	// is_optparam -
	//   Predicate: true if is optional input parameter (e.g. OptParam<...>)
	/////////////////////////
	template <typename T>
	struct is_optparam
	{
		static constexpr bool value = has_trait<T, OptParam>::value;
	};


	/////////////////////////
	// not_deferred -
	//   Predicate: true if underlying type is NOT deferred (e.g. OutParam<ImageRef<float>>)
	/////////////////////////
	template <typename T>
	struct not_deferred
	{
		static constexpr bool value = !is_deferred<T>::value;
	};


	/////////////////////////
	// true_pred -
	//   Predicate: always true
	/////////////////////////
	template <typename T>
	struct true_pred : std::true_type {};

	/////////////////////////
	// false_pred -
	//   Predicate: always false
	/////////////////////////
	template <typename T>
	struct false_pred: std::false_type {};
};

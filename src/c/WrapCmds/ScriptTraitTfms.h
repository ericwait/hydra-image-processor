#pragma once

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


	//////////////
	// Data type conversions (compile-time) for use with script trait types

	/////////////////////////
	// dtype_to_script -
	//   Transforms from data-trait type to script base-type
	//   (e.g. Image<int> -> Script::ArrayType*)
	/////////////////////////
	template <typename Traits>
	struct dtype_to_script {};

	template <template <typename> class DataTrait, typename BaseType>
	struct dtype_to_script<DataTrait<BaseType>>
	{
		using type = Script::ObjectType*;
	};

	template <typename BaseType>
	struct dtype_to_script<Script::Image<BaseType>>
	{
		using type = Script::ArrayType*;
	};

	template <typename BaseType>
	struct dtype_to_script<Script::ImageRef<BaseType>>
	{
		using type = Script::ArrayType*;
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
		using type = force_const_t<typename dtype_to_script<DataTraits>::type>;
	};

	template <typename DataTraits>
	struct iotrait_to_script<Script::OutParam<DataTraits>>
	{
		// TODO: Wrap pointer in simple unique object to avoid leaking on errors
		using type = typename dtype_to_script<DataTraits>::type;
	};

	/////////////////////////
	// dtype_to_concrete -
	//   Transforms from data-trait type to concrete base-type
	//   (e.g. Image<int> -> ImageContainer<int>)
	/////////////////////////
	template <typename Traits>
	struct dtype_to_concrete {};

	template <typename BaseType>
	struct dtype_to_concrete<Scalar<BaseType>>
	{
		using type = BaseType;
	};

	template <>
	struct dtype_to_concrete<Scalar<DeferredType>>
	{
		using type = Script::ObjectType*;
	};

	template <typename BaseType>
	struct dtype_to_concrete<Vector<BaseType>>
	{
		using type = Vec<BaseType>;
	};

	template <>
	struct dtype_to_concrete<Vector<DeferredType>>
	{
		using type = Script::ObjectType*;
	};

	template <typename BaseType>
	struct dtype_to_concrete<Image<BaseType>>
	{
		using type = ImageOwner<BaseType>;
	};

	template <>
	struct dtype_to_concrete<Image<DeferredType>>
	{
		using type = Script::ArrayType*;
	};

	template <typename BaseType>
	struct dtype_to_concrete<ImageRef<BaseType>>
	{
		using type = ImageView<BaseType>;
	};

	template <>
	struct dtype_to_concrete<ImageRef<DeferredType>>
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
		using type = typename dtype_to_concrete<DataTrait<BaseType>>::type;
	};

	template <template <typename> class IOTrait, template <typename> class DataTrait>
	struct iotrait_to_concrete<IOTrait<DataTrait<DeferredType>>>
	{
		using type = typename iotrait_to_script<IOTrait<DataTrait<DeferredType>>>::type;
	};

	/////////////////////////
	// deferred_dtype_to_concrete -
	//   Transforms all data-trait types to concrete types
	//   NOTE: The deferred types are all converted using the extra T parameter
	/////////////////////////
	template <typename Traits, typename T>
	struct deferred_dtype_to_concrete {};

	template <template <typename> class DataTrait, typename BaseType, typename T>
	struct deferred_dtype_to_concrete<DataTrait<BaseType>, T>
	{
		using type = typename dtype_to_concrete<DataTrait<BaseType>>::type;
	};

	template <template <typename> class DataTrait, typename T>
	struct deferred_dtype_to_concrete<DataTrait<DeferredType>, T>
	{
		using type = typename dtype_to_concrete<DataTrait<T>>::type;
	};


	template <typename Traits, typename T>
	struct deferred_iotrait_to_concrete_impl {};

	template <template <typename> class IOTrait, typename DataTraits, typename T>
	struct deferred_iotrait_to_concrete_impl<IOTrait<DataTraits>, T>
	{
		using type = typename deferred_dtype_to_concrete<DataTraits, T>::type;
	};

	/////////////////////////
	// deferred_to_concrete -
	//   Transforms all io-trait types to concrete types
	//   NOTE: The deferred types are all converted using the extra T parameter
	/////////////////////////
	template <typename T>
	struct deferred_to_concrete
	{
		template <typename Traits>
		struct tfm
		{
			using type = typename deferred_iotrait_to_concrete_impl<Traits, T>::type;
		};
	};

};

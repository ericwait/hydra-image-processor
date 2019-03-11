#pragma once

#include "Cuda/ImageView.h"

#include "mph/tuple_helpers.h"
#include "mph/qualifier_helpers.h"
#include "mph/const_string.h"

#include "ScriptCmds/ScriptTraits.h"
#include "ScriptCmds/ScriptTraitTfms.h"
#include "ScriptCmds/ArgConverter.h"

#include <tuple>
#include <string>
#include <cstdio>
#include <memory>
#include <type_traits>

namespace Script
{
	/////////////
	// MexArgConverter - Argument converter/loader for Matlab script engine
	template <typename Derived, typename... Layout>
	struct MexArgConverter: public ArgConverter<Derived, Layout...>
	{
		using BaseConverter = ArgConverter<Derived, Layout...>;

		using typename BaseConverter::ArgError;

		// Argument type layout alias (e.g. std::tuple<OutParam<Image<Deferred>>,...>)
		using typename BaseConverter::ArgLayout;

		// Script argument type layout (e.g. std::tuple<const PyArrayObject*,...>
		using typename BaseConverter::ScriptTypes;
		using typename BaseConverter::ScriptPtrs;

		// Concrete type layouts (e.g. std::tuple<PyObject*,...>)
		using typename BaseConverter::ArgTypes;

		// IO-type stripped layout subsets (e.g. OutParam<Image<Deferred>> -> Image<Deferred>)
		using typename BaseConverter::OutTypeLayout;
		using typename BaseConverter::InTypeLayout;
		using typename BaseConverter::OptTypeLayout;

		// IO Selectors
		using typename BaseConverter::OutputSel;
		using typename BaseConverter::InOptSel;
		using typename BaseConverter::InputSel;


		static void load(ScriptPtrs& scriptPtrs, mwSize nlhs, Script::ArrayType* plhs[], mwSize nrhs, const Script::ArrayType* prhs[])
		{
			// Validate input/output counts
			// If any outputs are expected we require at least one
			if ( nlhs == 0 && OutputSel::seq::size() > 0 )
				throw ArgError("Invalid number of output arguments: expected at least one output");

			if ( nlhs > OutputSel::seq::size() )
				throw ArgError("Invalid number of output arguments: at most %d outputs allowed", OutputSel::seq::size());

			if ( nrhs < InputSel::seq::size() )
				throw ArgError("Invalid number of input arguments: %d required inputs", InputSel::seq::size());

			if ( nrhs > InOptSel::seq::size() )
				throw ArgError("Invalid number of input arguments: at most %d inputs allowed", InOptSel::seq::size());

			// Linkup output subset
			loadSelected(typename OutputSel::seq{}, scriptPtrs, nlhs, plhs);

			// InOpt subset
			// NOTE: Will set scriptPtrs to null if they are past end of prhs
			//   nullptrs are ignored during conversion, allowing optionals
			loadSelected(typename InOptSel::seq{}, scriptPtrs, nrhs, prhs);
		}


		static void store(ScriptPtrs& scriptPtrs, mwSize nlhs, Script::ArrayType* plhs[], mwSize nrhs, const Script::ArrayType* prhs[])
		{}

	public:
		//////////////////////////////////
		// Basic type conversions
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
		{}

	private:
		template <std::size_t... Isel, std::size_t... Icnt>
		static void loadSelected_impl(mph::index_sequence<Isel...>, mph::index_sequence<Icnt...>, ScriptPtrs& scriptPtrs, mwSize countArgs, const Script::ArrayType* args[])
		{
			(void)std::initializer_list<int>
			{
				// NOTE: Have to be careful to only hook up &args[i] for i < count
				((std::get<Isel>(scriptPtrs) = (Icnt<countArgs) ? (&args[Icnt]) : (nullptr)), void(), 0)...
			};
		}

		template <std::size_t... Isel, std::size_t... Icnt>
		static void loadSelected_impl(mph::index_sequence<Isel...>, mph::index_sequence<Icnt...>, ScriptPtrs& scriptPtrs, mwSize countArgs, Script::ArrayType* args[])
		{
			(void)std::initializer_list<int>
			{
				// NOTE: Have to be careful to only hook up &args[i] for i < count
				((std::get<Isel>(scriptPtrs) = (Icnt<countArgs) ? (&args[Icnt]) : (nullptr)), void(), 0)...
			};
		}

		// Only hook up selected subset of outputs/input pointers to arg set
		template <typename ScriptArg, std::size_t... Isel>
		static void loadSelected(mph::index_sequence<Isel...>, ScriptPtrs& scriptPtrs, mwSize countArgs, ScriptArg&& scriptArgs)
		{
			loadSelected_impl(mph::index_sequence<Isel...>{}, mph::make_index_sequence<sizeof...(Isel)>{}, scriptPtrs, countArgs, std::forward<ScriptArg>(scriptArgs));
		}
	};
};

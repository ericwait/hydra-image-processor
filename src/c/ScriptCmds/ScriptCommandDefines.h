#pragma once

#include "mph/qualifier_helpers.h"

// Helper macro definitions for script commands and io mappers

// valid_map - Used in error check for valid OutMap templates
// NOTE: See _SCR_CHK_MAIN_TYPE for details
template <template <typename> class T, typename = void>
struct valid_map: std::false_type {};

template <template <typename> class T>
struct valid_map<T, mph::void_t<typename T<void>::type>>: std::true_type {};

#define _SCR_CHK_MAIN_TYPE(Name)								\
	template <typename T> struct OutMap_Impl;					\
	static_assert(valid_map<OutMap_Impl>::value,				\
			"HIP_COMPILE: Default IO type map undefined for: "	\
			#Name ". Command name may be mismatched");



/////////////
// SCR_DEFINE_IO_TYPE_MAP - Define an input->output io map for the specified script command
#define SCR_DEFINE_IO_TYPE_MAP(Name, OutT,InT)						\
	namespace CudaCall_##Name##_Stub								\
	{																\
		_SCR_CHK_MAIN_TYPE(Name)									\
		template <> struct OutMap_Impl<InT> {using type = OutT;};	\
	};

/////////////
// SCR_COMMAND_CLASSDEF - Create the class definition line for a script command
//   See ScrCmd*.h files for examples
#define SCR_COMMAND_CLASSDEF(Name) class ScriptCommand_##Name: public ScriptCommand_##Name##_Base


/////////////
// SCR_HELP_STRING - Add the command help string to a script command class definition
//   See ScrCmd*.h files for examples
#define SCR_HELP_STRING(Str)	using HelpStrType = decltype(mph::literal(Str));\
								static constexpr auto helpStr = mph::literal(Str)

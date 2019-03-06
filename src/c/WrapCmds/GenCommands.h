#include "ScriptTraits.h"
#include "LinkageTraitTfms.h"
#include "mph/preproc_helper.h"

#include <cstddef>

// Script parameters
#define SCR_PARAMS(...) __VA_ARGS__

// Main generator arguments passed to FOREACH to generate templates and argument information
#define SCR_OUTPUT(TypeMacro, VarName) (SCR_OUT_TRAIT(TypeMacro), VarName, nullptr)
#define SCR_INPUT(TypeMacro, VarName) (SCR_IN_TRAIT(TypeMacro), VarName, nullptr)
#define SCR_OPTIONAL(TypeMacro, VarName, DefVal) (SCR_OPT_TRAIT(TypeMacro), VarName, DefVal)

// Script argument IO traits
#define SCR_OUT_TRAIT(TypeMacro) Script::OutParam<TypeMacro>
#define SCR_IN_TRAIT(TypeMacro) Script::InParam<TypeMacro>
#define SCR_OPT_TRAIT(TypeMacro) Script::OptParam<TypeMacro>

// Script argument types
#define SCR_SCALAR(VarType) Script::Scalar<VarType>
#define SCR_VECTOR(VarType) Script::Vector<VarType>
#define SCR_IMAGE(VarType) Script::ImageRef<VarType>
#define SCR_IMAGE_CONVERT(VarType) Script::Image<VarType>

// Dynamic type (C++ type is inferred from input type)
#define SCR_DYNAMIC Script::DeferredType


// Expansion wrapper so that things like Params can be passed into sub-macros
#define _PASSTHRU(...) __VA_ARGS__
#define _STRINGIFY_IMPL(Name) mph::make_const_str(#Name)
#define _STRINGIFY(...) _STRINGIFY_IMPL(__VA_ARGS__)

// FOREACH selectors to expand subsets of the information in the script command definition
#define _SCR_PRM_TYPE_SEL(...) PRP_FOREACH(PRP_SEL3(1,0,0), __VA_ARGS__)
#define _SCR_PRM_NAME_SEL(...) PRP_FOREACH(PRP_SEL3(0,1,0), __VA_ARGS__)
#define _SCR_PRM_DEFVAL_SEL(...) PRP_FOREACH(PRP_SEL3(0,0,1), __VA_ARGS__)

#define _SCR_PRM_STRNAME_SEL(...) PRP_FOREACH(_STRINGIFY, PRP_FOREACH_C(PRP_SEL3(0,1,0), __VA_ARGS__))



#if defined(GENERATE_SCRIPT_COMMANDS)
// Generate static subclasses of ScriptCommand and ArgConverter for each command name
	#define SCR_BEGIN_COMMANDS
	#define SCR_END_COMMANDS

	#if defined(PY_BUILD)
		#define ARG_CONVERTER Script::PyArgConverter
	#elif defined(MEX_BUILD)
		#define ARG_CONVERTER Script::MexArgConverter
	#endif

	// Main definition of a script command (no automated process<OutT,InT>)
	#define SCR_CMD_NOPROC(Name, Params)				\
		_SCR_GEN_ARGCONVERTER(Name, _PASSTHRU(Params))	\
		_SCR_GEN_CMDCLASS_BASE_NOPROC(Name)

	// Main definition of a script command (Automatically call CudaFunc through stub)
	#define SCR_CMD(Name, Params, CudaFunc)				\
		_SCR_GEN_ARGCONVERTER(Name, _PASSTHRU(Params))	\
		_SCR_GEN_CMDCLASS_BASE(Name, CudaFunc)

	////////////////////
	// Helper generators for pieces of script command classes
	#define _SCR_GEN_ARGCONVERTER(Name, Params)					\
		template <typename InT> struct OutMap_##Name;			\
		struct ScriptCommand_##Name##_Parser					\
			: public ARG_CONVERTER<ScriptCommand_##Name##_Parser, _SCR_PRM_TYPE_SEL(Params)> \
		{														\
			template <typename InT>								\
			using OutMap = typename CudaCall_##Name##_Stub::OutMap<InT>;	\
																\
			inline static void setOptional(OptPtrs optPtrs)		\
			{													\
				mph::tuple_deref(optPtrs) = OptionalSel::select(std::make_tuple(_SCR_PRM_DEFVAL_SEL(Params)));	\
			}													\
																\
			using NameTuple = decltype(std::make_tuple(_SCR_PRM_STRNAME_SEL(Params)));	\
			static constexpr const NameTuple argNames{ _SCR_PRM_STRNAME_SEL(Params) };	\
		};


	#define _SCR_GEN_CMDCLASS_BASE_NOPROC(Name)			\
		class ScriptCommand_##Name;						\
		struct ScriptCommand_##Name##_Base				\
			: public ScriptCommandImpl<ScriptCommand_##Name, ScriptCommand_##Name##_Parser>	\
		{																		\
			inline static constexpr const char* commandName() { return #Name; }	\
		};


	#define _SCR_GEN_CMDCLASS_BASE(Name, CudaFunc)		\
		class ScriptCommand_##Name;						\
		struct ScriptCommand_##Name##_Base				\
			: public ScriptCommandImpl<ScriptCommand_##Name, ScriptCommand_##Name##_Parser>	\
		{																		\
			struct ProcessFunc													\
			{																	\
				template <typename... Args>										\
				inline static void run(Args&&... args)							\
				{ CudaCall_##Name##_Stub::CudaFunc##_stub(args...); }			\
			};																	\
			inline static constexpr const char* commandName() { return #Name; }	\
		};

#elif defined(GENERATE_CONSTEXPR_MEM)
// C++11 requires constexpr memory to be declared in a single .cpp module
	#define SCR_BEGIN_COMMANDS
	#define SCR_END_COMMANDS

	#define SCR_CMD_NOPROC(Name, Params) _SCR_GEN_CONSTEXP_MEM(Name)
	#define SCR_CMD(Name, Params, CudaFunc) _SCR_GEN_CONSTEXP_MEM(Name)

	#define _SCR_GEN_CONSTEXP_MEM(Name)	\
		constexpr const ScriptCommand_##Name##_Parser::NameTuple ScriptCommand_##Name##_Parser::argNames;	\
		constexpr ScriptCommand_##Name::HelpStrType ScriptCommand_##Name::helpStr;
		

#elif defined(GENERATE_DEFAULT_IO_MAPPERS)
// Generate InputType -> OutputType mapping (can be manually overloaded in ScripTypeMaps.h)
	#define SCR_BEGIN_COMMANDS
	#define SCR_END_COMMANDS

	#define SCR_CMD_NOPROC(Name, Params) _SCR_GEN_IOMAPPER(Name)
	#define SCR_CMD(Name, Params, CudaFunc) _SCR_GEN_IOMAPPER(Name)


	#define _SCR_GEN_IOMAPPER(Name)									\
		namespace CudaCall_##Name##_Stub							\
		{															\
			template <typename InT>									\
			struct OutMap_Impl { using type = InT;};				\
																	\
			template <template <typename> class T, typename BaseT>	\
			struct OutMap_Impl<T<BaseT>>							\
			{														\
				using type = T<typename OutMap_Impl<BaseT>::type>;	\
			};														\
																	\
			template <typename InT>									\
			using OutMap = typename OutMap_Impl<InT>::type;			\
		};

#elif defined(GENERATE_PROC_STUB_PROTOTYPES) || defined(GENERATE_PROC_STUBS)
// Generates the prototypes/implementation that link script engine frontends to cuda backend	
	#define SCR_BEGIN_COMMANDS
	#define SCR_END_COMMANDS

	#if defined(GENERATE_PROC_STUB_PROTOTYPES)
		#define _SCR_GEN_STUB(Name, Params, CudaFunc) _SCR_GEN_STUB_PROTO(Name, _PASSTHRU(Params), CudaFunc)
	#else
		#define _SCR_GEN_STUB(Name, Params, CudaFunc) _SCR_GEN_STUB_IMPL(Name, _PASSTHRU(Params), CudaFunc)
	#endif
	// Don't generate anything for _NOPROC commands, their stubs are created manually
	#define SCR_CMD_NOPROC(Name, Params)
	#define SCR_CMD(Name, Params, CudaFunc) _SCR_GEN_STUB(Name, _PASSTHRU(Params), CudaFunc)

	#define _APPLY_DEFER_TYPE(InType) PRP_CAT(_APPLY_DEF_TYPE_, InType)
	
	#define _APPLY_DEF_TYPE_bool(_traits, _varname) CudaLibrary::link_tfm<_traits,OutMap<bool>,bool> _varname
	#define _APPLY_DEF_TYPE_uint8_t(_traits, _varname) CudaLibrary::link_tfm<_traits,OutMap<uint8_t>,uint8_t> _varname
	#define _APPLY_DEF_TYPE_uint16_t(_traits, _varname) CudaLibrary::link_tfm<_traits,OutMap<uint16_t>,uint16_t> _varname
	#define _APPLY_DEF_TYPE_int16_t(_traits, _varname) CudaLibrary::link_tfm<_traits,OutMap<int16_t>,int16_t> _varname
	#define _APPLY_DEF_TYPE_uint32_t(_traits, _varname) CudaLibrary::link_tfm<_traits,OutMap<uint32_t>,uint32_t> _varname
	#define _APPLY_DEF_TYPE_int32_t(_traits, _varname) CudaLibrary::link_tfm<_traits,OutMap<int32_t>,int32_t> _varname
	#define _APPLY_DEF_TYPE_float(_traits, _varname) CudaLibrary::link_tfm<_traits,OutMap<float>,float> _varname
	#define _APPLY_DEF_TYPE_double(_traits, _varname) CudaLibrary::link_tfm<_traits,OutMap<double>,double> _varname

	#define _DROP_DEFVAL(_traits,_varname,_defval) _traits,_varname
	#define _SCR_PRM_DROP_DEFVAL(...) PRP_FOREACH_C(_DROP_DEFVAL, __VA_ARGS__)

	#define _SCR_PRM_SIGNATURE(InType, ...) PRP_FOREACH(_APPLY_DEFER_TYPE(InType), _SCR_PRM_DROP_DEFVAL(__VA_ARGS__))

	// TODO: Is there a better way to deal with _API prefix?
	#define _SCR_GEN_TYPED_PROTO(InType, Params, CudaFunc)			\
		IMAGE_PROCESSOR_API void CudaFunc##_stub(_SCR_PRM_SIGNATURE(InType, _PASSTHRU(Params)));

	#define _SCR_GEN_TYPED_IMPL(InType, Params, CudaFunc)			\
		void CudaFunc##_stub(_SCR_PRM_SIGNATURE(InType, _PASSTHRU(Params)))	\
		{															\
			CudaFunc(_SCR_PRM_NAME_SEL(Params));					\
		}	
	

	// This is the prototype stub call that separates the frontend(script)/backend(cuda) modules
	#define _SCR_GEN_STUB_PROTO(Name, Params, CudaFunc)					\
		namespace CudaCall_##Name##_Stub								\
		{																\
			_SCR_GEN_TYPED_PROTO(bool, _PASSTHRU(Params), CudaFunc)		\
			_SCR_GEN_TYPED_PROTO(uint8_t, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_PROTO(uint16_t, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_PROTO(int16_t, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_PROTO(uint32_t, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_PROTO(int32_t, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_PROTO(float, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_PROTO(double, _PASSTHRU(Params), CudaFunc)	\
		};

	// This is the implementation stub call that separates the frontend(script)/backend(cuda) modules
	#define _SCR_GEN_STUB_IMPL(Name, Params, CudaFunc)					\
		namespace CudaCall_##Name##_Stub								\
		{																\
			_SCR_GEN_TYPED_IMPL(bool, _PASSTHRU(Params), CudaFunc)		\
			_SCR_GEN_TYPED_IMPL(uint8_t, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_IMPL(uint16_t, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_IMPL(int16_t, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_IMPL(uint32_t, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_IMPL(int32_t, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_IMPL(float, _PASSTHRU(Params), CudaFunc)	\
			_SCR_GEN_TYPED_IMPL(double, _PASSTHRU(Params), CudaFunc)	\
		};

#elif defined(GENERATE_COMMAND_MAP)
// Generate list of commands used by python module loader and mex Info/Help commands
	#define SCR_BEGIN_COMMANDS const ScriptCommand::CommandList ScriptCommand::m_commands = {
	#define SCR_END_COMMANDS };

#define SCR_CMD_NOPROC(Name, Params) _SCR_CMD_MAP_LINE(Name)
#define SCR_CMD(Name, Params, CudaFunc) _SCR_CMD_MAP_LINE(Name)

#define _SCR_CMD_MAP_LINE(Name)				\
	{										\
		#Name,								\
		{&ScriptCommand_##Name::dispatch,	\
		&ScriptCommand_##Name::usage,		\
		&ScriptCommand_##Name::help,		\
		&ScriptCommand_##Name::info}		\
	},

#else
// Don't generate code for any other inclusions
	// Begin/End script command defs (necessary for list builder)
	#define SCR_BEGIN_COMMANDS
	#define SCR_END_COMMANDS

	// Main definition of a script command
	#define SCR_CMD_NOPROC(Name, Params)
	#define SCR_CMD(Name, Params, CudaFunc)
#endif

// Script Command list
#include "ScriptCommands.h"

// Undefine generator-specific tokens
//   Script-command class generators
#undef ARG_CONVERTER
#undef _SCR_GEN_ARGCONVERTER
#undef _SCR_GEN_CMDCLASS_BASE_NOPROC
#undef _SCR_GEN_CMDCLASS_BASE
//   Script-command class constexpr mem generators
#undef _SCR_GEN_CONSTEXP_MEM
//   Script-command in->out mapper generator
#undef _SCR_GEN_IOMAPPER
//   Script-to-cuda stub functions
#undef _SCR_GEN_STUB
#undef _SCR_GEN_TYPED_PROTO
#undef _SCR_GEN_STUB_IMPL


// Undefine the param generators
#undef SCR_OUT_TRAIT
#undef SCR_IN_TRAIT
#undef SCR_OPT_TRAIT
#undef SCR_SCALAR
#undef SCR_VECTOR
#undef SCR_IMAGE
#undef SCR_IMAGE_CONVERT
#undef SCR_DYNAMIC

// Undefine all general script command/param info
#undef SCR_BEGIN_COMMANDS
#undef SCR_END_COMMANDS
#undef SCR_PARAMS
#undef SCR_OUTPUT
#undef SCR_INPUT
#undef SCR_OPTIONAL
#undef SCR_CMD_NOPROC
#undef SCR_CMD

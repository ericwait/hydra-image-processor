#include "mph/preproc_helper.h"

// Main generator arguments passed to FOREACH to generate templates and argument information
#define SCR_OUTPUT(TypeMacro, VarName) (SCR_OUT_TRAIT(TypeMacro), VarName, nullptr)
#define SCR_INPUT(TypeMacro, VarName) (SCR_IN_TRAIT(TypeMacro), VarName, nullptr)
#define SCR_OPTIONAL(TypeMacro, VarName, DefVal) (SCR_OPT_TRAIT(TypeMacro), VarName, DefVal)


// Expansion wrapper so that things like Params can be passed into sub-macros
#define _PASSTHRU(...) __VA_ARGS__
#define _STRINGIFY_IMPL(Name) #Name
#define _STRINGIFY(...) _STRINGIFY_IMPL(__VA_ARGS__)

// FOREACH selectors to expand subsets of the information in the script command definition
#define _SCR_PRM_TYPE_SEL(...) PRP_FOREACH(PRP_SEL3(1,0,0), __VA_ARGS__)
#define _SCR_PRM_DEFVAL_SEL(...) PRP_FOREACH(PRP_SEL3(0,0,1), __VA_ARGS__)
#define _SCR_PRM_PROTO_SEL(...) PRP_FOREACH(PRP_SEL3(1,1,0), __VA_ARGS__)

#define _SCR_PRM_STRNAME_SEL(...) PRP_FOREACH(_STRINGIFY, PRP_FOREACH_C(PRP_SEL3(0,1,0), __VA_ARGS__))


#if defined(GENERATE_SCRIPT_COMMANDS)
// Generate static subclasses of ScriptCommand and ArgParser for each command name
	#if defined(PY_BUILD)
		#define ARG_CONVERTER Script::PyArgParser
	#elif defined(MEXBUILD)
		#define ARG_CONVERTER Script::MexArgParser
	#endif

	// Script parameters
	#define SCR_PARAMS(...) __VA_ARGS__
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

	// Main definition of a script command (no automated process<OutT,InT>)
	#define SCR_CMD_NOPROC(Name, Params)				\
		_SCR_GEN_ARGCONVERTER(Name, _PASSTHRU(Params))	\
		_SCR_GEN_CMDCLASS_BASE_NOPROC(Name)

	// Main definition of a script command (Automatically calls WrapperFunc)
	#define SCR_CMD(Name, Params, WrapperFunc)			\
		_SCR_GEN_ARGCONVERTER(Name, _PASSTHRU(Params))	\
		_SCR_GEN_CMDCLASS_BASE(Name, WrapperFunc)

	#define _SCR_SEL_TYPE(Params)

	////////////////////
	// Helper generators for pieces of script command classes
	#define _SCR_GEN_ARGCONVERTER(Name, Params)					\
		template <typename InT> struct OutMap_##Name;			\
		struct ScriptCommand_##Name##_Parser					\
			: public ARG_CONVERTER<ScriptCommand_##Name##_Parser, _SCR_PRM_TYPE_SEL(Params)> \
		{														\
			template <typename InT>								\
			using OutMap = typename OutMap_##Name<InT>::type;	\
																\
			inline static const char* argName(int idx);			\
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


	#define _SCR_GEN_CMDCLASS_BASE(Name, WrapperFunc)				\
		class ScriptCommand_##Name;									\
		struct ScriptCommand_##Name##_Base							\
			: public ScriptCommandImpl<ScriptCommand_##Name, ScriptCommand_##Name##_Parser>	\
		{																			\
			struct ProcessFunc														\
			{																		\
				template <typename... Args>											\
				inline static void run(Args&&... args) { WrapperFunc(args...); }	\
			};																		\
			inline static constexpr const char* commandName() { return #Name; }		\
		};

#elif defined(GENERATE_CONSTEXPR_MEM)
	#define SCR_CMD_NOPROC(Name, Params) _SCR_GEN_CONSTEXP_MEM(Name)
	#define SCR_CMD(Name, Params, WrapperFunc) _SCR_GEN_CONSTEXP_MEM(Name)

	#define _SCR_GEN_CONSTEXP_MEM(Name)	\
		constexpr const ScriptCommand_##Name##_Parser::NameTuple ScriptCommand_##Name##_Parser::argNames;
		

#elif defined(GENERATE_DEFAULT_IO_MAPPERS)
	#define SCR_CMD_NOPROC(Name, Params) _SCR_GEN_IOMAPPER(Name)
	#define SCR_CMD(Name, Params, WrapperFunc) _SCR_GEN_IOMAPPER(Name)


	#define _SCR_GEN_IOMAPPER(Name)									\
		template <typename InT>										\
		struct OutMap_##Name {using type = InT;};					\
		template <template <typename> class T, typename BaseT>		\
		struct OutMap_##Name<T<BaseT>>								\
		{															\
			using type = T<typename OutMap_##Name<BaseT>::type>;	\
		};															\
																	\
		template <typename InT>										\
		using OutMap_##Name##_T = typename OutMap_##Name<InT>::type;

#elif defined(GENERATE_COMMAND_MAP)
// Generate list of commands used by python module loader and mex passthrough Info/Help commands

#else
// Don't generate code for any other inclusions
	// Begin/End script command defs (necessary for list builder)
	#define SCR_BEGIN_COMMANDS
	#define SCR_END_COMMANDS

	// Script parameters
	#define SCR_PARAMS(...)
	// Script argument IO traits
	#define SCR_OUT_TRAIT(TypeMacro)
	#define SCR_IN_TRAIT(TypeMacro)
	#define SCR_OPT_TRAIT(TypeMacro)

	// Script argument types
	#define SCR_SCALAR(VarType)
	#define SCR_VECTOR(VarType)
	#define SCR_IMAGE(VarType)
	#define SCR_IMAGE_CONVERT(VarType)

	// Dynamic type (C++ type is inferred from input type)
	#define SCR_DYNAMIC

	// Main definition of a script command
	#define SCR_CMD_NOPROC(Name, Params)
	#define SCR_CMD(Name, Params, WrapperFunc)
#endif


#if defined(GEN_INCLUDE_FILE)
	#include "ScriptCommands.h"
#endif

SCR_CMD(Test, SCR_PARAMS
	(
		SCR_INPUT(SCR_IMAGE(SCR_DYNAMIC), imageIn),
		SCR_OUTPUT(SCR_IMAGE(SCR_DYNAMIC), imageOut),
		SCR_INPUT(SCR_IMAGE_CONVERT(float), kernel),
		SCR_OPTIONAL(SCR_SCALAR(int), numIterations, 1),
		SCR_OPTIONAL(SCR_SCALAR(int), device, -1)
	),
	closure
)

#define SCR_DEFINE_IO_TYPE_MAP(Name, OutT,InT)					\
	template <> struct OutMap_##Name<InT> {using type = OutT;};


#undef SCR_BEGIN_COMMANDS
#undef SCR_END_COMMANDS
#undef SCR_PARAMS
#undef SCR_OUTPUT
#undef SCR_INPUT
#undef SCR_OPTIONAL
#undef SCR_OUT_TRAIT
#undef SCR_IN_TRAIT
#undef SCR_OPT_TRAIT
#undef SCR_SCALAR
#undef SCR_VECTOR
#undef SCR_IMAGE
#undef SCR_IMAGE_CONVERT
#undef SCR_DYNAMIC
#undef SCR_CMD_NOPROC
#undef SCR_CMD

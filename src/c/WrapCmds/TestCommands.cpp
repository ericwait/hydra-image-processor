#include "ScriptCommandImpl.h"
#include "PyArgConverter.h"

#include "mph/tuple_helpers.h"
#include "Cuda/CWrappers.h"

#include <typeinfo>

#pragma optimize("",off)

namespace mph
{
	template <typename... Ts> using void_t = void;

	template <template <typename> class T, typename = void>
	struct valid_map: std::false_type {};

	template <template <typename> class T>
	struct valid_map<T, mph::void_t<T<void>>>: std::true_type {};
};

// TODO: Make a defines helper for this
#define _SCR_CHK_MAIN_TYPE(Name)								\
	static_assert(mph::valid_map<OutMap>::value,				\
			"HIP_COMPILE: Default IO type map undefined for: "	\
			#Name ". Command name may be mismatched");

#define SCR_DEFINE_IO_TYPE_MAP(Name, OutT,InT)						\
	namespace CudaCall_##Name##_Stub								\
	{																\
		_SCR_CHK_MAIN_TYPE(Name)									\
		template <> struct OutMap_Impl<InT> {using type = OutT;};	\
	};

// Goes in the io-typemap header
#define GENERATE_DEFAULT_IO_MAPPERS
#include "GenCommands.h"
#undef GENERATE_DEFAULT_IO_MAPPERS

SCR_DEFINE_IO_TYPE_MAP(Closure, int, int)

#define GENERATE_PROC_STUB_PROTOTYPES
#include "GenCommands.h"
#undef GENERATE_PROC_STUB_PROTOTYPES

#define GENERATE_PROC_STUBS
#include "GenCommands.h"
#undef GENERATE_PROC_STUBS



// Sequence of generators in main script commands header
#define GENERATE_SCRIPT_COMMANDS
#include "GenCommands.h"
#undef GENERATE_SCRIPT_COMMANDS


// 

#define SCR_COMMAND_CLASSDEF(Name) class ScriptCommand_##Name: public ScriptCommand_##Name##_Base


#define SCR_HELP_STRING(Str)	using HelpStrType = decltype(mph::literal(Str));\
								static constexpr auto helpStr = mph::literal(Str)
							 

#include "ScrCmdClosure.h"


SCR_COMMAND_CLASSDEF(Help)
{
public:
	SCR_HELP_STRING("Print detailed usage information for the specified command.");

	static void execute(const std::string& commandName)
	{
		if ( commandName.empty() )
			print_all();

		auto cmd = ScriptCommand::findCommand(commandName);
		if ( cmd )
			PySys_WriteStdout("%s\n", cmd->help().c_str());
	}

private:
	static void print_all()
	{
		ScriptCommand::CommandList cmds = ScriptCommand::commands();
		for ( const auto& it: cmds )
			PySys_WriteStdout("%s\n", it.second.usage().c_str());
	}

};


#define GENERATE_CONSTEXPR_MEM
#include "GenCommands.h"
#undef GENERATE_CONSTEXPR_MEM


#define GENERATE_COMMAND_MAP
#include "GenCommands.h"
#undef GENERATE_COMMAND_MAP


void testfunc()
{
	auto teststr = std::make_tuple(mph::literal("te"),mph::literal("st"));

	//static_assert(std::get<1>(teststr)[1]=='t', "****");

	//typename Script::filter<Script::is_outparam, std::tuple<Script::InParam<std::string>, Script::OutParam<std::vector<int>>>>::type t;
	//printf("%s\n", typeid(t).name());

	//Script::ArgConverter<Script::OutParam<int>, Script::InParam<int>>::OutArgs a;
	//printf("%s\n", typeid(a).name());

	//int a = mph::make_index_sequence<2000>::size();

	static_assert(Script::has_trait<Script::OutParam<int>, Script::OutParam>::value, "Uh oh");

	PyObject* out = nullptr;
	PyObject* in = nullptr;

	typename ScriptCommand_Help::ArgConverter::OutputSel::template type<ScriptCommand_Help::ArgConverter::ScriptPtrs> fdsf;

	//using ParseTupleArgs = typename mph::tuple_type_tfm<Script::PyParseType, ScriptCommandTest::Parser::InOptArgs>::type;
	//ParseTupleArgs parseVars;
	//auto varRefs = mph::tie_tuple(parseVars);
	//auto parseArgs = ScriptCommandTest::Parser::expand_parse_args(varRefs);
	//auto test = ScriptCommandTest::Parser::expand_parse_args_impl(varRefs, ScriptCommandTest::Parser::InOptTypeLayout(), mph::make_index_sequence<std::tuple_size<decltype(varRefs)>::value>());

	typename ScriptCommand_Closure::ArgConverter::S_InOpt<>::selector asdjfkl{};
	typename ScriptCommand_Closure::ArgConverter::OutTypeLayout jfls{};

	std::string tstjkfs = mph::literal("fjdklsfs");

	//typename ScriptCommandTest::ArgConverter::ArgPtrs fdd;
	//auto ate = ScriptCommandTest::DeferredSel::select(fdd);

	std::tuple<int,int,int> dff;
	//Script::arg_selector<mph::make_index_sequence<2>>::select(dff);

	mph::tuple_subset(mph::make_index_sequence<2>{}, (dff));
	mph::tuple_subset(mph::make_index_sequence<2>{}, std::make_tuple(1,2,3));
	//mph::internal::tuple_subset_impl(mph::make_index_sequence<2>{}, mph::tie_tuple(std::make_tuple(1, 2, 3)));
	
	typename ScriptCommand_Closure::ConcreteArgTypes<Vec<double>,Vec<float>> cctest;

	Script::TypeNameMap<int>::name();

	mph::tuple_fill_value(dff, 0);

	typename ScriptCommand_Closure::OptionalSel t1{};
	typename ScriptCommand_Closure::DeferredSel t2{};
	typename ScriptCommand_Closure::NondeferredSel t3{};
	typename ScriptCommand_Closure::NondeferInOptSel t4{};

	out = ScriptCommand_Closure::dispatch(nullptr, in);
	auto outHelp = ScriptCommand_Help::dispatch(nullptr, nullptr);

	std::string test = ScriptCommand_Closure::usage();

	ScriptCommand_Closure::ArgConverter::ArgLayout testy;

	ScriptCommand_Closure::OutMap<float> tres;

	std::tuple<PyObject const*> tmpOut{nullptr};
	auto arg_ref = mph::tie_tuple(tmpOut);
	
	std::tuple<PyObject const**> tst = 
		Script::PyParseExpand<Script::Scalar<float>>::argTuple(&std::get<0>(arg_ref));

	PyArg_ParseTuple(nullptr, "O", std::get<0>(tst));

}

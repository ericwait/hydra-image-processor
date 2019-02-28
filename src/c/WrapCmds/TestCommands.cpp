#include "ScriptCommand.h"
#include "PyConverters.h"

#include "mph/tuple_helpers.h"
#include "Cuda/CWrappers.h"

#include <typeinfo>

// Goes in the io-typemap header
#define GENERATE_DEFAULT_IO_MAPPERS
#include "GenCommands.h"
#undef GENERATE_DEFAULT_IO_MAPPERS

SCR_DEFINE_IO_TYPE_MAP(Test, int, int)

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

template <typename T, typename = void>
struct valid_type: std::false_type {};

template <typename T>
struct valid_type<T, typename T::ArgParser>
	: std::true_type {};


#define SCR_HELP_STRING(Str)	using HelpStrType = decltype(mph::literal(Str));\
								static constexpr auto helpStr = mph::literal(Str)
							 
class ScriptCommand_Test: public ScriptCommand_Test_Base
{
public:
	SCR_HELP_STRING(
		"jhfkldjakfhejkshlv\n"
		"fjhukehalkwehklufh\n");
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

	//Script::ArgParser<Script::OutParam<int>, Script::InParam<int>>::OutArgs a;
	//printf("%s\n", typeid(a).name());

	static_assert(Script::has_trait<Script::OutParam<int>, Script::OutParam>::value, "Uh oh");

	PyObject* out = nullptr;
	PyObject* in = nullptr;

	//using ParseTupleArgs = typename mph::tuple_type_tfm<Script::PyParseType, ScriptCommandTest::Parser::InOptArgs>::type;
	//ParseTupleArgs parseVars;
	//auto varRefs = mph::tie_tuple(parseVars);
	//auto parseArgs = ScriptCommandTest::Parser::expand_parse_args(varRefs);
	//auto test = ScriptCommandTest::Parser::expand_parse_args_impl(varRefs, ScriptCommandTest::Parser::InOptTypeLayout(), mph::make_index_sequence<std::tuple_size<decltype(varRefs)>::value>());

	typename ScriptCommand_Test::ArgParser::S_InOpt<>::selector asdjfkl{};
	typename ScriptCommand_Test::ArgParser::OutTypeLayout jfls{};

	std::string tstjkfs = mph::literal("fjdklsfs");

	//typename ScriptCommandTest::ArgParser::ArgPtrs fdd;
	//auto ate = ScriptCommandTest::DeferredSel::select(fdd);

	std::tuple<int,int,int> dff;
	//Script::arg_selector<mph::make_index_sequence<2>>::select(dff);

	mph::tuple_subset(mph::make_index_sequence<2>{}, (dff));
	mph::tuple_subset(mph::make_index_sequence<2>{}, std::make_tuple(1,2,3));
	//mph::internal::tuple_subset_impl(mph::make_index_sequence<2>{}, mph::tie_tuple(std::make_tuple(1, 2, 3)));
	
	typename ScriptCommand_Test::ConcreteArgTypes<Vec<double>,Vec<float>> cctest;

	Script::TypeNameMap<int>::name();

	mph::tuple_fill_value(dff, 0);

	typename ScriptCommand_Test::OptionalSel t1{};
	typename ScriptCommand_Test::DeferredSel t2{};
	typename ScriptCommand_Test::NondeferredSel t3{};
	typename ScriptCommand_Test::NondeferInOptSel t4{};

	out = ScriptCommand_Test::dispatch(nullptr, in);

	std::string test = ScriptCommand_Test::usage();

	ScriptCommand_Test::ArgParser::ArgLayout testy;

	ScriptCommand_Test::OutMap<float> tres;

	std::tuple<PyObject const*> tmpOut{nullptr};
	auto arg_ref = mph::tie_tuple(tmpOut);
	
	std::tuple<PyObject const**> tst = 
		Script::ParserArg<Script::Scalar<float>>::argTuple(&std::get<0>(arg_ref));

	PyArg_ParseTuple(nullptr, "O", std::get<0>(tst));

}

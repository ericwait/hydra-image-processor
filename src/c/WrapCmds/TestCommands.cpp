#include "ScriptCommand.h"
#include "PyConverters.h"

#include "mph/tuple_helpers.h"

#include <typeinfo>

struct TestFunc
{
};

#define DEF_CMD(Name, Params) \
	struct ScriptCommand_##Name##Parser : public Script::PyArgParser<ScriptCommand_##Name##Parser,Params> \
	{													\
		static constexpr const char* argName(int idx);	\
		static std::string inoptString();				\
		static std::string outputString();				\
	};

DEF_CMD(Test, SCR_PARAMS(SCR_OUTPUT(SCR_IMAGE_DYNAMIC(imOut)), SCR_OPTIONAL(SCR_VECTOR(sigmas, double), Vec<double>(1.0)), SCR_INPUT(SCR_IMAGE_DYNAMIC(im)), SCR_INPUT(SCR_IMAGE_CONVERT(kernel, float))))

template <typename T, typename = void>
struct valid_type: std::false_type {};

template <typename T>
struct valid_type<T, typename T::ArgParser>
	: std::true_type {};


class ScriptCommandTest;
class ScriptCommandTest_Route: public ScriptCommandImpl<ScriptCommandTest, ScriptCommand_TestParser>
{
public:
	static const char* commandName() { return "Test"; }
};

class ScriptCommandTest: public ScriptCommandTest_Route
{
public:
	SCR_DEFINE_IO_TYPE_MAP(int,float);

	using ArgLayout = ArgParser::ArgLayout;

	static typename ArgParser::OptArgs defaults();

	using Test = OutMap<int>;

	//static void execute(PyArrayObject*& imOut, Vec<double>& sigmas, const PyArrayObject*& im, ImageContainer<float>& kernel)
	//{
	//}

	//template <typename OutT, typename InT>
	//static void process(Script::ArrayType*& imOut, Vec<double>& sigmas, const Script::ArrayType*& im, ImageContainer<float>& kernel)
	//{
	//}
};

//DEFAULT_IO_TYPE_MAP(ScriptCommandTest);

inline typename ScriptCommandTest::ArgParser::OptArgs ScriptCommandTest::defaults()
{
	return ArgParser::OptionalSel::select(std::make_tuple(nullptr, Vec<double>(1.0), nullptr, nullptr));
}


const std::unordered_map<std::string, ScriptCommand::DispatchPtr> ScriptCommand::m_commands =
	{{ "Test",&ScriptCommandTest::dispatch }};


void testfunc()
{
	auto teststr = std::make_tuple("te","st");

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

	typename ScriptCommandTest::ArgParser::S_InOpt<>::selector asdjfkl{};
	typename ScriptCommandTest::ArgParser::OutTypeLayout jfls{};

	//typename ScriptCommandTest::ArgParser::ArgPtrs fdd;
	//auto ate = ScriptCommandTest::DeferredSel::select(fdd);

	using TestSel = typename Script::compose_selector<ScriptCommandTest::ArgLayout, Script::is_deferred>::selector;

	std::tuple<int,int,int> dff;
	//Script::arg_selector<mph::make_index_sequence<2>>::select(dff);

	mph::tuple_subset(mph::make_index_sequence<2>{}, (dff));
	mph::tuple_subset(mph::make_index_sequence<2>{}, std::make_tuple(1,2,3));
	//mph::internal::tuple_subset_impl(mph::make_index_sequence<2>{}, mph::tie_tuple(std::make_tuple(1, 2, 3)));
	

	typename ScriptCommandTest::OptionalSel t1{};
	typename ScriptCommandTest::DeferredSel t2{};
	typename ScriptCommandTest::NondeferredSel t3{};
	typename ScriptCommandTest::NondeferInOptSel t4{};

	out = ScriptCommandTest::dispatch(nullptr, in);
	ScriptCommandTest::defaults();

	ScriptCommandTest::ArgParser::ArgLayout testy;

	ScriptCommandTest::OutMap<float> tres;

	ScriptCommandTest::ArgParser::InArgs t;

	std::tuple<PyObject const*> tmpOut{nullptr};
	auto arg_ref = mph::tie_tuple(tmpOut);
	
	std::tuple<PyObject const**> tst = 
		Script::ParserArg<Script::Scalar<float>>::argTuple(&std::get<0>(arg_ref));

	PyArg_ParseTuple(nullptr, "O", std::get<0>(tst));

}

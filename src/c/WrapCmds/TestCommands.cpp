#include "ScriptCommand.h"
#include "PyConverters.h"

#include "mph/tuple_helpers.h"

#include <typeinfo>

DEFAULT_IO_TYPE_MAP(ScriptCommandTest);

using test_ap = SCR_PARAMS(SCR_OUTPUT(SCR_IMAGE_DYNAMIC(imOut)), SCR_OPTIONAL(SCR_VECTOR(sigmas,double),Vec<double>(1.0)), SCR_INPUT(SCR_IMAGE_DYNAMIC(im)), SCR_INPUT(SCR_IMAGE_CONVERT(kernel, float)));
class ScriptCommandTest: public ScriptCommandImpl<ScriptCommandTest, test_ap>
{
public:
	using TEST = GET_OUT_TYPE(ScriptCommandTest, uint8_t);
	using ArgLayout = Parser::Layout;

	static typename Parser::OptArgs defaults();

	//static void execute(PyObject*& imOut, Vec<double>& sigmas, const PyObject*& im, ImageContainer<float>& kernel)
	//{
	//}
};

typename ScriptCommandTest::Parser::OptArgs ScriptCommandTest::defaults()
{
	return mph::tuple_subset(typename Parser::opt_idx_seq(), std::make_tuple(nullptr, Vec<double>(1.0), nullptr, nullptr));
}


const std::unordered_map<std::string, ScriptCommand::DispatchPtr> ScriptCommand::m_commands =
	{{ "Test",&ScriptCommandTest::dispatch }};


void testfunc()
{
	//typename Script::filter<Script::is_outparam, std::tuple<Script::InParam<std::string>, Script::OutParam<std::vector<int>>>>::type t;
	//printf("%s\n", typeid(t).name());

	//Script::ArgParser<Script::OutParam<int>, Script::InParam<int>>::OutArgs a;
	//printf("%s\n", typeid(a).name());

	PyObject* out = nullptr;
	PyObject* in = nullptr;

	//using ParseTupleArgs = typename mph::tuple_type_tfm<Script::PyParseType, ScriptCommandTest::Parser::InOptArgs>::type;
	//ParseTupleArgs parseVars;
	//auto varRefs = mph::tie_tuple(parseVars);
	//auto parseArgs = ScriptCommandTest::Parser::expand_parse_args(varRefs);
	//auto test = ScriptCommandTest::Parser::expand_parse_args_impl(varRefs, ScriptCommandTest::Parser::InOptTypeLayout(), mph::make_index_sequence<std::tuple_size<decltype(varRefs)>::value>());


	out = ScriptCommandTest::dispatch(nullptr, in);
	ScriptCommandTest::defaults();

	ScriptCommandTest::Parser::Layout testy;

	ScriptCommandTest::Parser::inopt_im_idx_seq ims;
	ScriptCommandTest::Parser::inopt_im_defer_idx_seq a;
	ScriptCommandTest::Parser::out_defer_idx_seq b;

	ScriptCommandTest::Parser::InArgs t;

	std::tuple<PyObject const*> tmpOut{nullptr};
	auto arg_ref = mph::tie_tuple(tmpOut);
	
	std::tuple<PyObject const**> tst = 
		Script::ParserArg<Script::Scalar<float>>::argTuple(std::get<0>(arg_ref));

	PyArg_ParseTuple(nullptr, "O", std::get<0>(tst));

}

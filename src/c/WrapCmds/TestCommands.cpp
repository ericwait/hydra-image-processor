#include "ScriptCommand.h"
#include "PyConverters.h"

#include "mph/tuple_helpers.h"
#include "Cuda/CWrappers.h"

#include <typeinfo>

#define DEF_CMD_AUTO(Name, Params, CudaFunc) \
	struct ScriptCommand_##Name##_Parser : public Script::PyArgParser<ScriptCommand_##Name##_Parser,Params> \
	{															\
		inline static constexpr const char* argName(int idx);	\
		inline static std::string inoptString();				\
		inline static std::string outputString();				\
		inline static void setOptional(OptPtrs optPtrs);		\
	};															\
																\
	class ScriptCommand_##Name;									\
	struct ScriptCommand_##Name##_Base: public ScriptCommandImpl<ScriptCommand_##Name, ScriptCommand_##Name##_Parser>	\
	{															\
		struct ProcessFunc										\
		{														\
			template <typename... Args>							\
			inline static void run(Args&&... args) { CudaFunc(args...); }	\
		};																	\
		inline static constexpr const char* commandName() { return #Name; }	\
	};


DEF_CMD_AUTO(Test, SCR_PARAMS
	(
		SCR_INPUT(SCR_IMAGE(SCR_DYNAMIC), imageIn),
		SCR_OUTPUT(SCR_IMAGE(SCR_DYNAMIC), imageOut),
		SCR_INPUT(SCR_IMAGE_CONVERT(float), kernel),
		SCR_OPTIONAL(SCR_SCALAR(int), numIterations,1),
		SCR_OPTIONAL(SCR_SCALAR(int), device,-1)
	),
	closure
)

#undef DEF_CMD_AUTO

void ScriptCommand_Test_Parser::setOptional(ScriptCommand_Test_Parser::OptPtrs optPtrs)
{
	mph::tuple_deref(optPtrs) = OptionalSel::select(std::make_tuple(nullptr,nullptr,nullptr,1,-1));
}

template <typename T, typename = void>
struct valid_type: std::false_type {};

template <typename T>
struct valid_type<T, typename T::ArgParser>
	: std::true_type {};

class ScriptCommand_Test: public ScriptCommand_Test_Base
{
public:
////	//SCR_DEFAULT_IO_TYPE_MAP;
////
////	using ArgLayout = ArgParser::ArgLayout;
////
////	static typename ArgParser::OptArgs defaults();
////
////	using Test = OutMap<int>;
////	static_assert(std::is_same<uint8_t, OutMap<uint8_t>>::value, "");
////	static_assert(std::is_same<uint16_t, OutMap<uint16_t>>::value, "");
////	static_assert(std::is_same<int16_t, OutMap<int16_t>>::value, "");
////	static_assert(std::is_same<uint32_t, OutMap<uint32_t>>::value, "");
////	static_assert(std::is_same<int32_t, OutMap<int32_t>>::value, "");
////	static_assert(std::is_same<float, OutMap<float>>::value, "");
////	static_assert(std::is_same<double, OutMap<double>>::value, "");
////
////	//static void execute(const Script::ArrayType*& imageIn, Script::ArrayType*& imageOut, const ImageView<float>& kernel)
////	//{
////	//}
////
////	//template <typename OutT, typename InT>
////	//static void process(const Script::ArrayType*& imageIn, Script::ArrayType*& imageOut, const ImageView<float>& kernel, int numIterations, int device)
////	//{
////	//}
};


const std::unordered_map<std::string, ScriptCommand::DispatchPtr> ScriptCommand::m_commands =
{ { "Test",&ScriptCommand_Test::dispatch } };


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

	typename ScriptCommand_Test::ArgParser::S_InOpt<>::selector asdjfkl{};
	typename ScriptCommand_Test::ArgParser::OutTypeLayout jfls{};

	//typename ScriptCommandTest::ArgParser::ArgPtrs fdd;
	//auto ate = ScriptCommandTest::DeferredSel::select(fdd);

	Script::TypeToIdMap<char*>::typeId;

	std::tuple<int,int,int> dff;
	//Script::arg_selector<mph::make_index_sequence<2>>::select(dff);

	mph::tuple_subset(mph::make_index_sequence<2>{}, (dff));
	mph::tuple_subset(mph::make_index_sequence<2>{}, std::make_tuple(1,2,3));
	//mph::internal::tuple_subset_impl(mph::make_index_sequence<2>{}, mph::tie_tuple(std::make_tuple(1, 2, 3)));
	
	typename ScriptCommand_Test::ConcreteArgTypes<Vec<double>,Vec<float>> cctest;

	Script::TypeNameMap<int>::name;

	mph::tuple_fill_value(dff, 0);

	typename ScriptCommand_Test::OptionalSel t1{};
	typename ScriptCommand_Test::DeferredSel t2{};
	typename ScriptCommand_Test::NondeferredSel t3{};
	typename ScriptCommand_Test::NondeferInOptSel t4{};

	out = ScriptCommand_Test::dispatch(nullptr, in);

	ScriptCommand_Test::ArgParser::ArgLayout testy;

	ScriptCommand_Test::OutMap<float> tres;

	std::tuple<PyObject const*> tmpOut{nullptr};
	auto arg_ref = mph::tie_tuple(tmpOut);
	
	std::tuple<PyObject const**> tst = 
		Script::ParserArg<Script::Scalar<float>>::argTuple(&std::get<0>(arg_ref));

	PyArg_ParseTuple(nullptr, "O", std::get<0>(tst));

}
#undef DEF_MEX_COMMAND
#undef BEGIN_MEX_COMMANDS
#undef END_MEX_COMMANDS

#if defined(INSTANCE_COMMANDS)
#define BEGIN_MEX_COMMANDS
#define END_MEX_COMMANDS
#define DEF_MEX_COMMAND(name) Mex##name _ginstMex##name;
#elif defined(BUILD_COMMANDS)
#define BEGIN_MEX_COMMANDS																								\
	MexCommand* const MexCommand::m_commands[] =												\
							{

#define END_MEX_COMMANDS																								\
							};																							\
							const size_t MexCommand::m_numCommands = sizeof(MexCommand::m_commands) / sizeof(MexCommand*);

#define DEF_MEX_COMMAND(name) &_ginstMex##name,
#else
#define BEGIN_MEX_COMMANDS
#define END_MEX_COMMANDS
#define DEF_MEX_COMMAND(name)																	\
class Mex##name : public MexCommand															\
	{																									\
	public:																								\
	virtual std::string check(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) const;	\
	virtual void execute(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])  const;		\
	\
	virtual void usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs)  const;	\
	virtual void help(std::vector<std::string>& helpLines) const;									\
	\
	Mex##name() :MexCommand(#name) {};															\
	};
#endif

BEGIN_MEX_COMMANDS
// These are default commands defined for all MEX routines.
DEF_MEX_COMMAND(Info)
DEF_MEX_COMMAND(Help)
// Additional specific mex commands should be added here.
DEF_MEX_COMMAND(DeviceCount)
DEF_MEX_COMMAND(DeviceStats)
DEF_MEX_COMMAND(Closure)
DEF_MEX_COMMAND(ElementWiseDifference)
DEF_MEX_COMMAND(EntropyFilter)
DEF_MEX_COMMAND(Gaussian)
DEF_MEX_COMMAND(GetMinMax)
DEF_MEX_COMMAND(HighPassFilter)
DEF_MEX_COMMAND(LoG)
DEF_MEX_COMMAND(MaxFilter)
DEF_MEX_COMMAND(MeanFilter)
DEF_MEX_COMMAND(MedianFilter)
DEF_MEX_COMMAND(MinFilter)
DEF_MEX_COMMAND(MultiplySum)
DEF_MEX_COMMAND(Opener)
DEF_MEX_COMMAND(StdFilter)
DEF_MEX_COMMAND(Sum)
DEF_MEX_COMMAND(WienerFilter)
END_MEX_COMMANDS

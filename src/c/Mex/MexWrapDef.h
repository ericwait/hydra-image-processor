#undef BEGIN_WRAP_COMMANDS
#undef END_WRAP_COMMANDS
#undef DEF_WRAP_COMMAND

#if defined(INSTANCE_COMMANDS)
#define BEGIN_WRAP_COMMANDS
#define END_WRAP_COMMANDS
#define DEF_WRAP_COMMAND(name) Mex##name _ginstMex##name;
#elif defined(BUILD_COMMANDS)
#define BEGIN_WRAP_COMMANDS																								\
	MexCommand* const MexCommand::m_commands[] =																		\
							{

#define END_WRAP_COMMANDS																								\
							};																							\
							const std::size_t MexCommand::m_numCommands = sizeof(MexCommand::m_commands) / sizeof(MexCommand*);

#define DEF_WRAP_COMMAND(name) &_ginstMex##name,
#else
#define BEGIN_WRAP_COMMANDS
#define END_WRAP_COMMANDS
#define DEF_WRAP_COMMAND(name)																\
class Mex##name : public MexCommand															\
{																							\
	public:																					\
	virtual std::string check(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) const;	\
	virtual void execute(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])  const;		\
	\
	virtual void usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs)  const;	\
	virtual void help(std::vector<std::string>& helpLines) const;									\
	\
	Mex##name() :MexCommand(#name) {};														\
};
#endif

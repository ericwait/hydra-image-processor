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
DEF_MEX_COMMAND(AddConstant)
DEF_MEX_COMMAND(AddImageWith)
DEF_MEX_COMMAND(ApplyPolyTransformation)
DEF_MEX_COMMAND(ContrastEnhancement)
DEF_MEX_COMMAND(DeviceCount)
DEF_MEX_COMMAND(DeviceStats)
DEF_MEX_COMMAND(EntropyFilter)
DEF_MEX_COMMAND(GaussianFilter)
DEF_MEX_COMMAND(Histogram)
DEF_MEX_COMMAND(ImagePow)
DEF_MEX_COMMAND(LinearUnmixing)
DEF_MEX_COMMAND(MarkovRandomFieldDenoiser)
DEF_MEX_COMMAND(MaxFilterEllipsoid)
DEF_MEX_COMMAND(MaxFilterKernel)
DEF_MEX_COMMAND(MaxFilterNeighborhood)
DEF_MEX_COMMAND(MeanFilter)
DEF_MEX_COMMAND(MedianFilter)
DEF_MEX_COMMAND(MinFilterEllipsoid)
DEF_MEX_COMMAND(MinFilterKernel)
DEF_MEX_COMMAND(MinFilterNeighborhood)
DEF_MEX_COMMAND(MinMax)
DEF_MEX_COMMAND(MorphologicalClosure)
DEF_MEX_COMMAND(MorphologicalOpening)
DEF_MEX_COMMAND(MultiplyImage)
DEF_MEX_COMMAND(MultiplyTwoImages)
DEF_MEX_COMMAND(NormalizedCovariance)
DEF_MEX_COMMAND(NormalizedHistogram)
DEF_MEX_COMMAND(OtsuThresholdFilter)
DEF_MEX_COMMAND(OtsuThresholdValue)
DEF_MEX_COMMAND(RegionGrowing)
DEF_MEX_COMMAND(Resize)
DEF_MEX_COMMAND(SumArray)
DEF_MEX_COMMAND(Segment)
DEF_MEX_COMMAND(StdFilter)
DEF_MEX_COMMAND(ThresholdFilter)
DEF_MEX_COMMAND(TileImage)
DEF_MEX_COMMAND(Variance)
END_MEX_COMMANDS
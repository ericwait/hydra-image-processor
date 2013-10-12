#include "mex.h"
#include <string>
#include "Vec.h"
#include "Process.h"
#include "MexCommand.h"

extern "C" 
{
	void mexCleanUp()
	{
		MexCommand::cleanUp();
	}
};

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (MexCommand::needsInit())
	{
		mexAtExit(mexCleanUp);
		MexCommand::init();
	}

	//TODO: print the actual error reason after the list
	if (nrhs==0)
		mexErrMsgTxt(MexCommand::printUsageList().c_str());

	if (!mxIsChar(prhs[0]))
		mexErrMsgTxt(MexCommand::printUsageList().c_str());

	char cmd[255];
	mxGetString(prhs[0],cmd,255);

	MexCommand* thisCommand = MexCommand::getCommand(cmd);
	if (thisCommand==NULL)
		mexErrMsgTxt(MexCommand::printUsageList().c_str());

	std::string errMsg = thisCommand->check(nlhs,plhs,nrhs-1,prhs+1);
	if (errMsg.length()!=0)
	{
		mexPrintf("%s\n",thisCommand->printUsage().c_str());
		mexErrMsgTxt(errMsg.c_str());
	}

	try
	{
		thisCommand->execute(nlhs,plhs,nrhs-1,prhs+1);
	}
	catch (std::string err)
	{
		mexErrMsgTxt(err.c_str());
	}
}
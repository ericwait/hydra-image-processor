#include "mex.h"
#include <string>
#include "Process.h"
#include "MexCommand.h"
#include "CWrappers.h"
 
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
 
 	if (nrhs==0)
 		mexErrMsgTxt(MexCommand::printUsageList().c_str());
 
 	if (!mxIsChar(prhs[0]))
 		mexErrMsgTxt(MexCommand::printUsageList().c_str());
 
 	char cmdIn[255];
 	mxGetString(prhs[0],cmdIn,255);
	char cmd[255];

	sprintf_s(cmd,"Mex%s",cmdIn);
 
 	MexCommand* thisCommand = MexCommand::getCommand(cmd);
 	if (thisCommand==NULL)
 		mexErrMsgTxt(MexCommand::printUsageList().c_str());

	if (nrhs>1 && mxIsChar(prhs[1]))
	{
		char buff[255];
		mxGetString(prhs[1],buff,255);
		if (_strcmpi("help",buff)==0)
		{
			mexPrintf("%s\n%s",thisCommand->printUsage().c_str(),thisCommand->printHelp().c_str());
			return;
		}
	}
 
 	std::string errMsg = thisCommand->check(nlhs,plhs,nrhs-1,prhs+1);
 	if (errMsg.length()!=0)
 	{
		mexPrintf("%s\n%s",thisCommand->printUsage().c_str(),thisCommand->printHelp().c_str());
 		mexErrMsgTxt(errMsg.c_str());
 	}
 
 	try
 	{
 		thisCommand->execute(nlhs,plhs,nrhs-1,prhs+1);
 	}
 	catch (std::string err)
 	{
		clearDevice();
 		mexErrMsgTxt(err.c_str());
 	}
	catch(std::logic_error err)
	{
		clearDevice();
		mexErrMsgTxt(err.what());
	}
	catch(std::runtime_error err)
	{
		clearDevice();
		mexErrMsgTxt(err.what());
	}
 }
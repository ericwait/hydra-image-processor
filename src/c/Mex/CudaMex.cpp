#include "MexCommand.h"

std::string MexCommand::mexName = "Mex";

void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
	MexCommand::run(nlhs,plhs,nrhs,prhs);

	return;
}
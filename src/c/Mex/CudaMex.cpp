#include "MexCommand.h"

void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
	MexCommand::run(nlhs,plhs,nrhs,prhs);

	return;
}
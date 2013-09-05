#include "mex.h"
#include <string>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs<1 || !mxIsChar(prhs[0])) mexErrMsgTxt("Usage:\n");

	char* command = mxArrayToString(prhs[0]);

	if (_strcmpi("load image",command)==0)
	{
		if (nrhs!=0 || nlhs!=1) mexErrMsgTxt("no right hand arguments or more than one left hand argument!\n");
		mxGetImagData(prhs[0]);

	}else if (_strcmpi("",command)==0)
	{

	}
}
#include "MexCommand.h"
#include "CWrappers.cuh"

void MexDeviceCount::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{

	plhs[0] = mxCreateDoubleScalar(deviceCount());
}

std::string MexDeviceCount::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nlhs != 1)
		return "Requires one output!";

	return "";
}

std::string MexDeviceCount::printUsage()
{
	return "numCudaDevice = CudaMex('DeviceCount');";
}

std::string MexDeviceCount::printHelp()
{
	std::string msg = "\tThis returns the number of CUDA capable devices are present.\n";
	msg += "\tUse this number to pass in an index to all other CudaMex commands.\n";
	msg += "\tDevice indices start at one. They then go up to and include the number returned here.\n";
	msg += "\n";
	return msg;
}
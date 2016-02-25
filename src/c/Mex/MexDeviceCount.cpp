#include "MexCommand.h"
#include "CWrappers.h"

void MexDeviceCount::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(nlhs==1)
	{
		plhs[0] = mxCreateDoubleScalar(deviceCount());
	}else if (nlhs>1)
	{
		size_t* memStats;
		int numDevices = memoryStats(&memStats);
		plhs[0] = mxCreateDoubleScalar(numDevices);
		const char* fieldNames[] = {"total","available"};
		mwSize dims[2] ={1,numDevices};
		plhs[1] = mxCreateStructArray(2,dims,2,fieldNames);

		int total_field = mxGetFieldNumber(plhs[1],"total");
		int avail_field = mxGetFieldNumber(plhs[1],"available");

		for(int i=0; i<numDevices; ++i)
		{
			mxSetFieldByNumber(plhs[1],i,total_field,mxCreateDoubleScalar(double(memStats[i*2])));
			mxSetFieldByNumber(plhs[1],i,avail_field,mxCreateDoubleScalar(double(memStats[i*2+1])));
		}
	}
}

std::string MexDeviceCount::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nlhs < 1)
		return "Requires one output!";

	return "";
}

std::string MexDeviceCount::printUsage()
{
	return "[numCudaDevices, [memoryStats]] = CudaMex('DeviceCount');";
}

std::string MexDeviceCount::printHelp()
{
	std::string msg = "\tThis returns the number of CUDA capable devices are present.\n";
	msg += "\tIf there is a second left hand argument, a structure with the total and available memory is also returned.\n";
	msg += "\tUse this number to pass in an index to all other CudaMex commands.\n";
	msg += "\tDevice indices start at one. They then go up to and include the number returned here.\n";
	msg += "\n";
	return msg;
}
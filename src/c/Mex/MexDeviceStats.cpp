#include "MexCommand.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/CudaDeviceStats.h"

void MexDeviceStats::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
    DevStats* devStats;
	mwSize numDevices = (mwSize)deviceStats(&devStats);

    mwSize dims[2] = {numDevices, 1};
    const char* fieldNames[] = {"name", "major", "minor", "constMem", "sharedMem", "totalMem", "tccDriver", "mpCount", "threadsPerMP", "warpSize", "maxThreads"};
    plhs[0] = mxCreateStructArray(2, dims, 11, fieldNames);
    int name_field = mxGetFieldNumber(plhs[0], "name");
    int major_field = mxGetFieldNumber(plhs[0], "major");
    int minor_field = mxGetFieldNumber(plhs[0], "minor");
    int constMem_field = mxGetFieldNumber(plhs[0], "constMem");
    int sharedMem_field = mxGetFieldNumber(plhs[0], "sharedMem");
    int totalMem_field = mxGetFieldNumber(plhs[0], "totalMem");
    int tccDriver_field = mxGetFieldNumber(plhs[0], "tccDriver");
    int mpCount_field = mxGetFieldNumber(plhs[0], "mpCount");
    int threadsPerMP_field = mxGetFieldNumber(plhs[0], "threadsPerMP");
    int warpSize_field = mxGetFieldNumber(plhs[0], "warpSize");
    int maxThreads_field = mxGetFieldNumber(plhs[0], "maxThreads");

    for(int device = 0; device<numDevices; ++device)
    {
        DevStats curDevice = devStats[device];

        mxSetFieldByNumber(plhs[0], device, name_field, mxCreateString(curDevice.name.c_str()));
        mxSetFieldByNumber(plhs[0], device, major_field, mxCreateDoubleScalar(curDevice.major));
        mxSetFieldByNumber(plhs[0], device, minor_field, mxCreateDoubleScalar(curDevice.minor));
        mxSetFieldByNumber(plhs[0], device, constMem_field, mxCreateDoubleScalar(curDevice.constMem));
        mxSetFieldByNumber(plhs[0], device, sharedMem_field, mxCreateDoubleScalar(curDevice.sharedMem));
        mxSetFieldByNumber(plhs[0], device, totalMem_field, mxCreateDoubleScalar(curDevice.totalMem));
        mxSetFieldByNumber(plhs[0], device, tccDriver_field, mxCreateLogicalScalar(curDevice.tccDriver));
        mxSetFieldByNumber(plhs[0], device, mpCount_field, mxCreateDoubleScalar(curDevice.mpCount));
        mxSetFieldByNumber(plhs[0], device, threadsPerMP_field, mxCreateDoubleScalar(curDevice.threadsPerMP));
        mxSetFieldByNumber(plhs[0], device, warpSize_field, mxCreateDoubleScalar(curDevice.warpSize));
        mxSetFieldByNumber(plhs[0], device, maxThreads_field, mxCreateDoubleScalar(curDevice.maxThreads));
    }

    delete[] devStats;
}

std::string MexDeviceStats::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
    if(nlhs<1)
        return "Requires one output!";

    return "";
}

void MexDeviceStats::usage(std::vector<std::string>& outArgs, std::vector<std::string>& inArgs) const
{
    outArgs.push_back("deviceStatsArray");
}

void MexDeviceStats::help(std::vector<std::string>& helpLines) const
{
    helpLines.push_back("This will return the statistics of each Cuda capable device installed.");

    helpLines.push_back("\tDeviceStatsArray -- this is an array of structs, one struct per device.");
    helpLines.push_back("\t\tThe struct has these fields: name, major, minor, constMem, sharedMem, totalMem, tccDriver, mpCount, threadsPerMP, warpSize, maxThreads.");
}

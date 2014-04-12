#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"

void MexMinMax::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>1)
		device = mat_to_c((int)mxGetScalar(prhs[1]));

	Vec<size_t> imageDims;
	HostPixelType* imageIn;
	CudaProcessBuffer cudaBuffer(device);
	setupImagePointers(prhs[0],&imageIn,&imageDims);

	HostPixelType minVal, maxVal;

	cudaBuffer.getMinMax(imageIn,imageDims.product(),minVal,maxVal);

	plhs[0] = mxCreateDoubleScalar(minVal);
	plhs[1] = mxCreateDoubleScalar(maxVal);
}

std::string MexMinMax::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<1 || nrhs>2)
		return "Incorrect number of inputs!";

	if (nlhs!=2)
		return "Requires two outputs!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	// 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	// 	if (numDims>3 || numDims<2)
	// 		return "Image can only be either 2D or 3D!";

	return "";
}

std::string MexMinMax::printUsage()
{
	return "[min max] = CudaMex('MinMax',imageIn,[device]);";
}

std::string MexMinMax::printHelp()
{
	std::string msg = "\tReturns the minimum and maximum values.\n";
	msg += "\n";
	return msg;
}
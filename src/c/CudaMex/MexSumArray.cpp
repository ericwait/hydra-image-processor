#include "MexCommand.h"
#include "CWrappers.cuh"

void MexSumArray::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>1)
		device = mat_to_c((int)mxGetScalar(prhs[1]));

	Vec<size_t> imageDims;
	HostPixelType* imageIn;
	CudaProcessBuffer cudaBuffer(device);
	setupImagePointers(prhs[0],&imageIn,&imageDims);

	double sm = cSumArray(imageIn,imageDims.product());

	plhs[0] = mxCreateDoubleScalar(sm);
}

std::string MexSumArray::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<1 || nrhs>2)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one outputs!";

// 	if (!mxIsUint8(prhs[0]))
// 		return "Image has to be formated as a uint8!";

// 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
// 	if (numDims>3 || numDims<2)
// 		return "Image can only be either 2D or 3D!";

	return "";
}

std::string MexSumArray::printUsage()
{
	return "sum = CudaMex('SumArray',imageIn,[device]);";
}

std::string MexSumArray::printHelp()
{
	std::string msg = "\tSums up all the values in the given image.\n";
	msg += "\n";
	return msg;
}
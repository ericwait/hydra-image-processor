#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"

void NormalizedCovariance::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	Vec<size_t> imageDims1, imageDims2;
	HostPixelType* imageIn1, * imageIn2, * imageOut;
	CudaProcessBuffer cudaBuffer(device);
	setupImagePointers(prhs[0],&imageIn1,&imageDims1);
	setupImagePointers(prhs[1],&imageIn2,&imageDims2);

	if (imageDims1!=imageDims2)
		mexErrMsgTxt("Image Dimensions Must Match!\n");

	double normCoVar = cudaBuffer.normalizedCovariance(imageIn1,imageIn2,imageDims1);

	plhs[0] = mxCreateDoubleScalar(normCoVar);
}

std::string NormalizedCovariance::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

// 	if (!mxIsUint8(prhs[0]) || !mxIsUint8(prhs[1]))
// 		return "Images has to be formated as a uint8!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3 || numDims<2)
		return "Image can only be either 2D or 3D!";

	numDims = mxGetNumberOfDimensions(prhs[1]);
	if (numDims>3 || numDims<2)
		return "Image can only be either 2D or 3D!";

	if (!mxIsDouble(prhs[2]))
		return "Factor needs to be a double!";

	return "";
}

std::string NormalizedCovariance::printUsage()
{
	return "normalizedCovariance = CudaMex('NormalizedCovariance',imageIn1,imageIn2,[device]);";
}

std::string NormalizedCovariance::printHelp()
{
	std::string msg = "\tThis will calculate how similar the images are to one another.\n";
	msg += "\tThe return value will be between [-1,1].  Where 1 is exactly the same and -1 is exactly the opposite.\n";
	msg += "\tImages must match in dimension.\n";
	msg += "\n";
	return msg;
}

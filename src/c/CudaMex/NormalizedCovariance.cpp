#include "MexCommand.h"

void NormalizedCovariance::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims1;
	HostPixelType* imageIn1;
	setupImagePointers(prhs[0],&imageIn1,&imageDims1);
	Vec<unsigned int> imageDims2;
	HostPixelType* imageIn2;
	setupImagePointers(prhs[1],&imageIn2,&imageDims2);

	if (imageDims1!=imageDims2)
		mexErrMsgTxt("Image Dimensions Must Match!\n");

	double normCoVar = normalizedCovariance(imageIn1,imageIn2,imageDims1);

	plhs[0] = mxCreateDoubleScalar(normCoVar);
}

std::string NormalizedCovariance::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]) || !mxIsUint8(prhs[1]))
		return "Images has to be formated as a uint8!";

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
	return "normalizedCovariance = CudaMex('NormalizedCovariance',imageIn1,imageIn2)";
}
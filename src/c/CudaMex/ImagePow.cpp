#include "MexCommand.h"
#include "Process.h"

void ImagePow::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<size_t> imageDims;
	ImageContainer* imageIn, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&plhs[0],&imageOut);

	double p = mxGetScalar(prhs[1]);
	imagePow(imageIn,imageOut,(int)p);

	delete imageIn;
	delete imageOut;
}

std::string ImagePow::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=2)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3 || numDims<2)
		return "Image can only be either 2D or 3D!";

	if (!mxIsDouble(prhs[1]))
		return "Power has to be a single double!";

	return "";
}

std::string ImagePow::printUsage()
{
	return "imageOut = CudaMex('ImagePow',imageIn,power)";
}
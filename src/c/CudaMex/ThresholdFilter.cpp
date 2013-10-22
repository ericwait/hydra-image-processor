#include "MexCommand.h"
#include "Process.h"

void ThresholdFilter::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	MexImagePixelType* imageIn, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

	double thresh = mxGetScalar(prhs[1]);

	thresholdFilter(imageIn,imageOut,imageDims,thresh);
}

std::string ThresholdFilter::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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
		return "Threshold needs to be a single double!";

	return "";
}

std::string ThresholdFilter::printUsage()
{
	return "imageOut = CudaMex('ThresholdFilter',imageIn,threshold)";
}
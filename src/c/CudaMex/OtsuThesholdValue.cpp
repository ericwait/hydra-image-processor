#include "MexCommand.h"

void OtsuThesholdValue::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	HostPixelType* imageIn;
	setupImagePointers(prhs[0],&imageIn,&imageDims);

	double thresh;

	thresh = (double)otsuThesholdValue(imageIn,imageDims);

	plhs[0] = mxCreateDoubleScalar(thresh);
}

std::string OtsuThesholdValue::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=1)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3 || numDims<2)
		return "Image can only be either 2D or 3D!";

	return "";
}

std::string OtsuThesholdValue::printUsage()
{
	return "threshold = CudaMex('OtsuThesholdValue',imageIn)";
}

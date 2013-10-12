#include "MexCommand.h"
#include "Process.h"

void CalculateMinMax::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	MexImagePixelType* imageIn;
	setupImagePointers(prhs[0],&imageIn,&imageDims);

	MexImagePixelType mn, mx;

	calculateMinMax(imageIn,imageDims,mn,mx);

	plhs[0] = mxCreateDoubleScalar((double)mn);
	plhs[1] = mxCreateDoubleScalar((double)mx);
}

std::string CalculateMinMax::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=1)
		return "Incorrect number of inputs!";

	if (nlhs!=2)
		return "Requires two outputs!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	int numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3 || numDims<2)
		return "Image can only be either 2D or 3D!";

	return "";
}

std::string CalculateMinMax::printUsage()
{
	return "[min max] = CudaMex('CalculateMinMax',imageIn)";
}
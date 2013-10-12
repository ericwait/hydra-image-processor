#include "MexCommand.h"
#include "Process.h"

void MaxFilter::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	MexImagePixelType* imageIn, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

	double* nbh = (double*)mxGetData(prhs[1]);

	Vec<unsigned int> neighborhood((unsigned int)nbh[0],(unsigned int)nbh[1],(unsigned int)nbh[2]);
	maxFilter(imageIn,imageOut,imageDims,neighborhood);
}

std::string MaxFilter::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=2)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	int numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3 || numDims<2)
		return "Image can only be either 2D or 3D!";

	if (!mxIsDouble(prhs[1]))
		return "Neighborhood needs to be an array of three doubles!";

	numDims = mxGetNumberOfDimensions(prhs[1]);
	if (numDims!=3)
		return "Neighborhood needs to be an array of three doubles!";

	return "";
}

std::string MaxFilter::printUsage()
{
	return "imageOut = CudaMex('MaxFilter',imageIn,[neighborhoodX,neighborhoodY,neighborhoodZ])";
}
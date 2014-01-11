#include "MexCommand.h"
#include "Process.h"

void MinFilterNeighborhood::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<size_t> imageDims;
	ImageContainer* imageIn, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&plhs[0],&imageOut);

	double* nbh = (double*)mxGetData(prhs[1]);
	Vec<size_t> neighborhood((size_t)nbh[0],(size_t)nbh[1],(size_t)nbh[2]);
	minFilter(imageIn,imageOut,neighborhood);

	delete imageIn;
	delete imageOut;
}

std::string MinFilterNeighborhood::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=2)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims>3 || imgNumDims<2)
		return "Image can only be either 2D or 3D!";

	size_t numEl = mxGetNumberOfElements(prhs[1]);
	if (numEl!=3)
		return "Neighborhood needs to be an array of three doubles!";

	return "";
}

std::string MinFilterNeighborhood::printUsage()
{
	return "imageOut = CudaMex('MinFilterNeighborhood',imageIn,[neighborhoodX,neighborhoodY,neighborhoodZ])";
}
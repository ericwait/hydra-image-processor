#include "MexCommand.h"
#include "Process.h"

void MinFilter::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	MexImagePixelType* imageIn, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

	double* neighborhoodD = (double*)mxGetData(prhs[1]);
	Vec<unsigned int> neighborhood((int)neighborhoodD[0],(int)neighborhoodD[1],(int)neighborhoodD[2]);

	minFilter(imageIn,imageOut,imageDims,neighborhood);
}

std::string MinFilter::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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

	size_t numEl= mxGetNumberOfElements(prhs[1]);
	if (numEl!=3 || !mxIsDouble(prhs[1]))
		return "Neighborhood has to be an array of three doubles!";

	return "";
}

std::string MinFilter::printUsage()
{
	return "imageOut = CudaMex('MinFilter',imageIn,[NeighborhoodX,NeighborhoodY,NeighborhoodZ])";
}
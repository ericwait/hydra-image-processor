#include "MexCommand.h"

void NormalizedHistogram::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	MexImagePixelType* imageIn;
	setupImagePointers(prhs[0],&imageIn,&imageDims);

	int arraySize;
	double* hist = retrieveNormalizedHistogram(imageIn,imageDims,arraySize);

	const mwSize DIM = arraySize;
	plhs[0] = mxCreateNumericArray(1,&DIM,mxDOUBLE_CLASS,mxREAL);
	double* histPr = mxGetPr(plhs[0]);

	memcpy(histPr,hist,sizeof(double)*arraySize);
}

std::string NormalizedHistogram::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=1)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one outputs!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3 || numDims<2)
		return "Image can only be either 2D or 3D!";

	return "";
}

std::string NormalizedHistogram::printUsage()
{
	return "histogram = CudaMex('NormalizedHistogram',imageIn)";
}
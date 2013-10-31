#include "MexCommand.h"

void Histogram::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<size_t> imageDims;
	ImageContainer* imageIn;
	setupImagePointers(prhs[0],&imageIn);

	int arraySize;
	size_t* hist = retrieveHistogram(imageIn,arraySize);

	const mwSize DIM = arraySize;
	plhs[0] = mxCreateNumericArray(1,&DIM,mxDOUBLE_CLASS,mxREAL);
	double* histPr = mxGetPr(plhs[0]);

	for (int i=0; i<arraySize; ++i)
		histPr[i] = (double)hist[i];
}

std::string Histogram::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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

std::string Histogram::printUsage()
{
	return "histogram = CudaMex('Histogram',imageIn)";
}
#include "MexCommand.h"
#include "Process.h"

void MultiplyImageWith::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<size_t> imageDims1;
	ImageContainer* imageIn1, * imageOut;
	HostPixelType* mexImageOut;
	setupImagePointers(prhs[0],&imageIn1,&plhs[0],&mexImageOut,&imageOut);
	Vec<size_t> imageDims2;
	ImageContainer* imageIn2;
	setupImagePointers(prhs[1],&imageIn2);

	multiplyImageWith(imageIn1,imageIn2,imageOut);
	rearange(imageOut,mexImageOut);

	delete imageIn1;
	delete imageIn2;
	delete imageOut;
}

std::string MultiplyImageWith::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=2)
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

	return "";
}

std::string MultiplyImageWith::printUsage()
{
		return "imageOut = CudaMex('MultiplyImageWith',imageIn1,imageIn2)";
}
#include "MexCommand.h"
#include "Process.h"

void Mask::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims1;
	HostPixelType* imageIn1, * imageOut;
	setupImagePointers(prhs[0],&imageIn1,&imageDims1,&plhs[0],&imageOut);
	Vec<unsigned int> imageDims2;
	HostPixelType* imageIn2;
	setupImagePointers(prhs[1],&imageIn2,&imageDims2);


	double threshold = 1;
	if (nrhs==3)
		threshold = mxGetScalar(prhs[2]);

	mask(imageIn1,imageIn2,imageOut,imageDims1,threshold);
}

std::string Mask::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=2 && nrhs!=3)
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

	if (nrhs==3)
	{
		if (!mxIsDouble(prhs[2]))
			return "Threshold needs to be a double!";
	}

	return "";
}

std::string Mask::printUsage()
{
	return "imageOut = CudaMex('Mask',imageIn,imageInMask[,threshold])";
}
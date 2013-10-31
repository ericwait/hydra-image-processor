#include "MexCommand.h"
#include "Process.h"

void AddConstant::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	Vec<size_t> imageDims;
	ImageContainer* imageIn, * imageOut;
	HostPixelType* mexImageOut;
	setupImagePointers(prhs[0],&imageIn,&plhs[0],&mexImageOut,&imageOut);

	double additive = mxGetScalar(prhs[1]);

	addConstant(imageIn,imageOut,additive);
	rearange(imageOut,mexImageOut);
}

std::string AddConstant::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
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
		return "Additive needs to be a single double!";

	return "";
}

std::string AddConstant::printUsage()
{
	return "imageOut = CudaMex('AddConstant',imageIn,additive)";
}
#include "MexCommand.h"
#include "CHelpers.h"

void MaxFilterCircle::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	MexImagePixelType* imageIn, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

	double radius = mxGetScalar(prhs[1]);
	Vec<unsigned int> kernDims;
	double* circleKernel = createCircleKernel((int)radius,kernDims);
	maxFilter(imageIn,imageOut,imageDims,kernDims,circleKernel);
}

std::string MaxFilterCircle::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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

	if (!mxIsDouble(prhs[1]))
		return "Radius needs to be a single double!";

	return "";
}

std::string MaxFilterCircle::printUsage()
{
	return "imageOut = CudaMex('MaxFilterCircle',imageIn,radius)";
}
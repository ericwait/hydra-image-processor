#include "MexCommand.h"
#include "Process.h"
#include "CHelpers.h"

void MorphOpening::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<size_t> imageDims;
	ImageContainer* imageIn, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&plhs[0],&imageOut);

	double* radiiD = (double*)mxGetData(prhs[1]);

	Vec<size_t> radii((size_t)radiiD[0],(size_t)radiiD[1],(size_t)radiiD[2]);
	Vec<size_t> kernDims;
	double* circleKernel = createEllipsoidKernel(radii,kernDims);
	morphOpening(imageIn,imageOut,kernDims,circleKernel);

	delete imageIn;
	delete imageOut;
	delete[] circleKernel;
}

std::string MorphOpening::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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
	if (numEl!=3 || !mxIsDouble(prhs[1]))
		return "Radii must be an array of three doubles!";

	return "";
}

std::string MorphOpening::printUsage()
{
	return "imageOut = CudaMex('MorphOpening',imageIn,[radiusX,radiusY,radiusZ])";
}
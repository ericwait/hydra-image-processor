#include "MexCommand.h"
#include "Process.h"


void ApplyPolyTransformation::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<size_t> imageDims;
	ImageContainer* imageIn, * imageOut;
	HostPixelType* mexImageOut;
	setupImagePointers(prhs[0],&imageIn,&plhs[0],&mexImageOut,&imageOut);

	double a = mxGetScalar(prhs[1]);
	double b = mxGetScalar(prhs[2]);
	double c = mxGetScalar(prhs[3]);

	applyPolyTransformation(imageIn,imageOut,a,b,c,std::numeric_limits<HostPixelType>::min(),std::numeric_limits<HostPixelType>::max());
	rearange(imageOut,mexImageOut);

	delete imageIn;
	delete imageOut;
}


std::string ApplyPolyTransformation::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=4)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3 || numDims<2)
		return "Image can only be either 2D or 3D!";

	if (!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) || !mxIsDouble(prhs[3]))
		return "a,b,c all have to be doubles!";

	return "";
}

std::string ApplyPolyTransformation::printUsage()
{
	return "imageOut = CudaMex('ApplyPolyTransformation',imageIn,a,b,c)";
}
#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"

void ApplyPolyTransformation::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{

	Vec<size_t> imageDims;
	HostPixelType* imageIn, * imageOut;
	CudaProcessBuffer cudaBuffer;
	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

	double a, b, c;
	double mn = (double)std::numeric_limits<HostPixelType>::min();
	double mx = (double)std::numeric_limits<HostPixelType>::max();
	a = mxGetScalar(prhs[1]);
	b = mxGetScalar(prhs[2]);
	c = mxGetScalar(prhs[3]);

	if (nrhs>5)
		mx = MIN(mxGetScalar(prhs[5]),mx);

	if (nrhs>4)
		mn = MAX(mxGetScalar(prhs[4]),mn);

	HostPixelType minVal = (HostPixelType)mn;
	HostPixelType maxVal = (HostPixelType)mx;

	cudaBuffer.applyPolyTransformation(imageIn,imageDims,a,b,c,minVal,maxVal,&imageOut);
}

std::string ApplyPolyTransformation::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<4 && 6<nrhs)
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
	return "imageOut = CudaMex('ApplyPolyTransformation',imageIn,a,b,c,[min],[max]);";
}

std::string ApplyPolyTransformation::printHelp()
{
	std::string msg = "\ta, b, and c are the polynomial curve parameters for the transfer function which maps imageIn to imageOut.\n";
	msg += "\tmin and max are optional clamping parameters that will clamp the output values between [min,max].\n";
	msg += "\tIf min and max are not supplied, imageOut is clamped to the range of imageIn's type.\n";
	msg += "\timageOut will be the same dimension as imageIn.\n";
	msg += "\n";
	return msg;
}

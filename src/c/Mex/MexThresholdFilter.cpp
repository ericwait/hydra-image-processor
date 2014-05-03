#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.cuh"

void MexThresholdFilter::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	double thresh = mxGetScalar(prhs[1]);

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		thresholdFilter(imageIn,imageDims,(unsigned char)thresh,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		thresholdFilter(imageIn,imageDims,(unsigned int)thresh,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		thresholdFilter(imageIn,imageDims,(int)thresh,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		thresholdFilter(imageIn,imageDims,(float)thresh,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		thresholdFilter(imageIn,imageDims,thresh,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexThresholdFilter::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<2 || nrhs>3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3)
		return "Image can have a maximum of three dimensions!";

	if (!mxIsDouble(prhs[1]))
		return "Threshold needs to be a single double!";

	return "";
}

std::string MexThresholdFilter::printUsage()
{
	return "imageOut = CudaMex('ThresholdFilter',imageIn,threshold,[device]);";
}

std::string MexThresholdFilter::printHelp()
{
	std::string msg = "\tMaps any value >= thresh to the max value of the image space.";
	msg += "\tAll other values will be set at the minimum of the image space.\n";
	msg += "\n";
	return msg;
}

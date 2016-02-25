#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexOtsuThresholdFilter::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	double alpha = 1.0;
	if (nrhs==2)
		alpha = mxGetScalar(prhs[1]);

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		otsuThresholdFilter(imageIn,imageDims,alpha,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		otsuThresholdFilter(imageIn,imageDims,alpha,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		otsuThresholdFilter(imageIn,imageDims,alpha,&imageOut,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		otsuThresholdFilter(imageIn,imageDims,alpha,&imageOut,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		otsuThresholdFilter(imageIn,imageDims,alpha,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		otsuThresholdFilter(imageIn,imageDims,alpha,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		otsuThresholdFilter(imageIn,imageDims,alpha,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexOtsuThresholdFilter::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<1 || nrhs>3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3)
		return "Image can have a maximum of three dimensions!";

	if (nrhs==2)
		if (!mxIsDouble(prhs[1]))
			return "Alpha needs to be a single double!";

	return "";
}

std::string MexOtsuThresholdFilter::printUsage()
{
	return "imageOut = CudaMex('OtsuThresholdFilter',imageIn,[alpha],[device]);";
}

std::string MexOtsuThresholdFilter::printHelp()
{
	std::string msg = "\tCalculates a two class threshold using Otsu's method.\n";
	msg += "\tEach pixel/voxel >= the threshold is set to the max value of the image space.";
	msg += "\tAll other values will be set at the minimum of the image space.\n";
	msg += "\n";
	return msg;
}


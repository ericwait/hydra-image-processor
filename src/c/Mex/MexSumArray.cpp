#include "MexCommand.h"
#include "CWrappers.cuh"
#include "Vec.h"

void MexSumArray::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>1)
		device = mat_to_c((int)mxGetScalar(prhs[1]));

	double sm;

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		sm = (double)sumArray(imageIn,imageDims.product(),device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		sm = (double)sumArray(imageIn,imageDims.product(),device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		sm = (double)sumArray(imageIn,imageDims.product(),device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		sm = (double)sumArray(imageIn,imageDims.product(),device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		sm = (double)sumArray(imageIn,imageDims.product(),device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		sm = sumArray(imageIn,imageDims.product(),device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		sm = sumArray(imageIn,imageDims.product(),device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}

	plhs[0] = mxCreateDoubleScalar(sm);
}

std::string MexSumArray::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<1 || nrhs>2)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one outputs!";

// 	if (!mxIsUint8(prhs[0]))
// 		return "Image has to be formated as a uint8!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3)
		return "Image can have a maximum of three dimensions!";

	return "";
}

std::string MexSumArray::printUsage()
{
	return "sum = CudaMex('SumArray',imageIn,[device]);";
}

std::string MexSumArray::printHelp()
{
	std::string msg = "\tSums up all the values in the given image.\n";
	msg += "\n";
	return msg;
}
#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.cuh"

void MexNormalizedCovariance::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	double normCoVar;

	Vec<size_t> imageDims1, imageDims2;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn1,* imageIn2;
		setupImagePointers(prhs[0],&imageIn1,&imageDims1);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims1!=imageDims2)
			mexErrMsgTxt("Image Dimensions Must Match!\n");

		normCoVar = normalizedCovariance(imageIn1,imageIn2,imageDims1,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn1,* imageIn2;
		setupImagePointers(prhs[0],&imageIn1,&imageDims1);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims1!=imageDims2)
			mexErrMsgTxt("Image Dimensions Must Match!\n");

		normCoVar = normalizedCovariance(imageIn1,imageIn2,imageDims1,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn1,* imageIn2;
		setupImagePointers(prhs[0],&imageIn1,&imageDims1);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims1!=imageDims2)
			mexErrMsgTxt("Image Dimensions Must Match!\n");

		normCoVar = normalizedCovariance(imageIn1,imageIn2,imageDims1,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn1,* imageIn2;
		setupImagePointers(prhs[0],&imageIn1,&imageDims1);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims1!=imageDims2)
			mexErrMsgTxt("Image Dimensions Must Match!\n");

		normCoVar = normalizedCovariance(imageIn1,imageIn2,imageDims1,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn1,* imageIn2;
		setupImagePointers(prhs[0],&imageIn1,&imageDims1);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims1!=imageDims2)
			mexErrMsgTxt("Image Dimensions Must Match!\n");

		normCoVar = normalizedCovariance(imageIn1,imageIn2,imageDims1,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn1,* imageIn2;
		setupImagePointers(prhs[0],&imageIn1,&imageDims1);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims1!=imageDims2)
			mexErrMsgTxt("Image Dimensions Must Match!\n");

		normCoVar = normalizedCovariance(imageIn1,imageIn2,imageDims1,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn1,* imageIn2;
		setupImagePointers(prhs[0],&imageIn1,&imageDims1);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims1!=imageDims2)
			mexErrMsgTxt("Image Dimensions Must Match!\n");

		normCoVar = normalizedCovariance(imageIn1,imageIn2,imageDims1,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}

	plhs[0] = mxCreateDoubleScalar(normCoVar);
}

std::string MexNormalizedCovariance::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t numDims1 = mxGetNumberOfDimensions(prhs[0]);
	if (numDims1>3)
		return "Image can have a maximum of three dimensions!";

	size_t numDims2= mxGetNumberOfDimensions(prhs[1]);
	if (numDims2>3)
		return "Image can have a maximum of three dimensions!";

	if (numDims1!=numDims2)
		return "Images must have the same dimensions!";

	return "";
}

std::string MexNormalizedCovariance::printUsage()
{
	return "normalizedCovariance = CudaMex('NormalizedCovariance',imageIn1,imageIn2,[device]);";
}

std::string MexNormalizedCovariance::printHelp()
{
	std::string msg = "\tThis will calculate how similar the images are to one another.\n";
	msg += "\tThe return value will be between [-1,1].  Where 1 is exactly the same and -1 is exactly the opposite.\n";
	msg += "\tImages must match in dimension.\n";
	msg += "\n";
	return msg;
}

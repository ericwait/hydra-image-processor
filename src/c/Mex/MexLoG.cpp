#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexLoG::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	int device = 0;

	if(nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	double* sigmasMat = (double*)mxGetData(prhs[1]);
	Vec<float> sigmas((float)(sigmasMat[0]), (float)(sigmasMat[1]), (float)(sigmasMat[2]));	

	Vec<size_t> imageDims;
	setupDims(prhs[0], &imageDims);
	float* imageOut = NULL;
	setupOutputPointers(&plhs[0], imageDims, &imageOut);

	if(mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn;
		setupInputPointers(prhs[0], &imageDims, &imageIn);

		loG(imageIn, imageDims, sigmas, &imageOut, device);
	} else if(mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn;
		setupInputPointers(prhs[0], &imageDims, &imageIn);

		loG(imageIn, imageDims, sigmas, &imageOut, device);
	} else if(mxIsInt16(prhs[0]))
	{
		short* imageIn;
		setupInputPointers(prhs[0], &imageDims, &imageIn);

		loG(imageIn, imageDims, sigmas, &imageOut, device);
	} else if(mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn;
		setupInputPointers(prhs[0], &imageDims, &imageIn);

		loG(imageIn, imageDims, sigmas, &imageOut, device);
	} else if(mxIsInt32(prhs[0]))
	{
		int* imageIn;
		setupInputPointers(prhs[0], &imageDims, &imageIn);

		loG(imageIn, imageDims, sigmas, &imageOut, device);
	} else if(mxIsSingle(prhs[0]))
	{
		float* imageIn;
		setupInputPointers(prhs[0], &imageDims, &imageIn);

		loG(imageIn, imageDims, sigmas, &imageOut, device);
	} else if(mxIsDouble(prhs[0]))
	{
		double* imageIn;
		setupInputPointers(prhs[0], &imageDims, &imageIn);

		loG(imageIn, imageDims, sigmas, &imageOut, device);
	} else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexLoG::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	if(nrhs<2||nrhs>3)
		return "Incorrect number of inputs!";

	if(nlhs!=1)
		return "Requires one output!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if(numDims>3)
		return "Image can have a maximum of three dimensions!";

	size_t numEl = mxGetNumberOfElements(prhs[1]);
	if(numEl!=3||!mxIsDouble(prhs[1]))
		return "Sigmas must be an array of three doubles!";

	return "";
}

void MexLoG::usage(std::vector<std::string>& outArgs, std::vector<std::string>& inArgs) const
{
	inArgs.push_back("imageIn");
	inArgs.push_back("sigma");
	inArgs.push_back("device");

	outArgs.push_back("imageOut");
}

void MexLoG::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("Smooths image using a Gaussian kernel.");

	helpLines.push_back("\tImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).");
	helpLines.push_back("\tSigma -- these values will create a n-dimensional Gaussian kernel to get a smoothed image that will be subtracted of the original.");
	helpLines.push_back("\tDevice -- this is an optional parameter that indicates which Cuda capable device to use.");

	helpLines.push_back("\tImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.");
}
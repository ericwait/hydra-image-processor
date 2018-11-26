#include "MexCommand.h"
#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"
#include "MexKernel.h"

void MexLoG::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	int device = -1;
	int numIterations = 1;

	if (!mxIsEmpty(prhs[2]))
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	double* sigmasMat = (double*)mxGetData(prhs[1]);
	Vec<double> sigmas(sigmasMat[0], sigmasMat[1], sigmasMat[2]);

	ImageDimensions imageDims;
	if (mxIsLogical(prhs[0]))
	{
		bool* imageInPtr;
		float* imageOutPtr;
		Script::setupInputPointers(prhs[0], imageDims, &imageInPtr);
		Script::setupOutputPointers(&plhs[0], imageDims, &imageOutPtr);

		ImageContainer<bool> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);

	}
	else if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageInPtr;
		float* imageOutPtr;
		Script::setupInputPointers(prhs[0], imageDims, &imageInPtr);
		Script::setupOutputPointers(&plhs[0], imageDims, &imageOutPtr);

		ImageContainer<unsigned char> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageInPtr;
		float* imageOutPtr;
		Script::setupInputPointers(prhs[0], imageDims, &imageInPtr);
		Script::setupOutputPointers(&plhs[0], imageDims, &imageOutPtr);

		ImageContainer<unsigned short> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageInPtr;
		float* imageOutPtr;
		Script::setupInputPointers(prhs[0], imageDims, &imageInPtr);
		Script::setupOutputPointers(&plhs[0], imageDims, &imageOutPtr);

		ImageContainer<short> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageInPtr;
		float* imageOutPtr;
		Script::setupInputPointers(prhs[0], imageDims, &imageInPtr);
		Script::setupOutputPointers(&plhs[0], imageDims, &imageOutPtr);

		ImageContainer<unsigned int> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageInPtr;
		float* imageOutPtr;
		Script::setupInputPointers(prhs[0], imageDims, &imageInPtr);
		Script::setupOutputPointers(&plhs[0], imageDims, &imageOutPtr);

		ImageContainer<int> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageInPtr;
		float* imageOutPtr;
		Script::setupInputPointers(prhs[0], imageDims, &imageInPtr);
		Script::setupOutputPointers(&plhs[0], imageDims, &imageOutPtr);

		ImageContainer<float> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageInPtr;
		float* imageOutPtr;
		Script::setupInputPointers(prhs[0], imageDims, &imageInPtr);
		Script::setupOutputPointers(&plhs[0], imageDims, &imageOutPtr);

		// TODO: Do we really want to use float outputs here?
		ImageContainer<double> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		LoG(imageIn, imageOut, sigmas, device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexLoG::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	if (nrhs != 3)
		return "Incorrect number of inputs!";

	if (nlhs != 1)
		return "Requires one output!";

	std::size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims > 5)
		return "Image can have a maximum of five dimensions!";

	if (!mxIsDouble(prhs[1]))
		return "Sigmas must a vector of three positive doubles!";

	if (!(mxGetNumberOfDimensions(prhs[1]) != 3))
		return "Sigmas must a vector of three positive doubles!";

	if (!mxIsEmpty(prhs[2]))
	{
		int numIter = (int)mxGetScalar(prhs[2]);
		if (numIter < 1)
			return "Number of iterations must be 1 or greater!";
	}


	return "";
}

void MexLoG::usage(std::vector<std::string>& outArgs, std::vector<std::string>& inArgs) const
{
	inArgs.push_back("arrayIn");
	inArgs.push_back("sigmas");
	inArgs.push_back("[device]");
	outArgs.push_back("arrayOut");
}

void MexLoG::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("Apply a Lapplacian of Gaussian filter with the given sigmas.");

	helpLines.push_back("\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.");
	helpLines.push_back("\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine");
	helpLines.push_back("\t\thow to stride or jump to the next spatial block.");
	helpLines.push_back("");

	helpLines.push_back("\tSigmas = This should be an array of three positive values that represent the standard deviation of a Gaussian curve.");
	helpLines.push_back("\t\tZeros (0) in this array will not smooth in that direction.");
	helpLines.push_back("");

	helpLines.push_back("\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.");
	helpLines.push_back("\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split");
	helpLines.push_back("\t\tthe data across multiple devices.");
	helpLines.push_back("");

	helpLines.push_back("\timageOut = This will be an array of the same type and shape as the input array.");
}

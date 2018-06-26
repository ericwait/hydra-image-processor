#include "MexCommand.h"
#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"
#include "MexKernel.h"

void MexMinMax::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	int device = -1;

	if (!mxIsEmpty(prhs[1]))
		device = mat_to_c((int)mxGetScalar(prhs[1]));

	ImageDimensions imageDims;
	double minOut = 0.0;
	double maxOut = 0.0;

	if (mxIsLogical(prhs[0]))
	{
		bool* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<bool> imageIn(imageInPtr, imageDims);

		bool lclMin = 0;
		bool lclMax = 0;
		minMax(imageIn, lclMin, lclMax, device);
		minOut = double(lclMin);
		maxOut = double(lclMax);
	}
	else if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<unsigned char> imageIn(imageInPtr, imageDims);

		unsigned char lclMin = 0;
		unsigned char lclMax = 0;
		minMax(imageIn, lclMin, lclMax, device);
		minOut = double(lclMin);
		maxOut = double(lclMax);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<unsigned short> imageIn(imageInPtr, imageDims);

		unsigned short lclMin = 0;
		unsigned short lclMax = 0;
		minMax(imageIn, lclMin, lclMax, device);
		minOut = double(lclMin);
		maxOut = double(lclMax);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<short> imageIn(imageInPtr, imageDims);

		short lclMin = 0;
		short lclMax = 0;
		minMax(imageIn, lclMin, lclMax, device);
		minOut = double(lclMin);
		maxOut = double(lclMax);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<unsigned int> imageIn(imageInPtr, imageDims);

		unsigned int lclMin = 0;
		unsigned int lclMax = 0;
		minMax(imageIn, lclMin, lclMax, device);
		minOut = double(lclMin);
		maxOut = double(lclMax);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<int> imageIn(imageInPtr, imageDims);

		int lclMin = 0;
		int lclMax = 0;
		minMax(imageIn, lclMin, lclMax, device);
		minOut = double(lclMin);
		maxOut = double(lclMax);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<float> imageIn(imageInPtr, imageDims);

		float lclMin = 0;
		float lclMax = 0;
		minMax(imageIn, lclMin, lclMax, device);
		minOut = double(lclMin);
		maxOut = double(lclMax);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<double> imageIn(imageInPtr, imageDims);

		double lclMin = 0;
		double lclMax = 0;
		minMax(imageIn, lclMin, lclMax, device);
		minOut = double(lclMin);
		maxOut = double(lclMax);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}

	plhs[0] = mxCreateDoubleScalar(minOut);
	if (nlhs>1)
		plhs[1] = mxCreateDoubleScalar(maxOut);
}

std::string MexMinMax::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	if (nrhs != 2)
		return "Incorrect number of inputs!";

	if (nlhs < 1)
		return "Requires one output!";

	if (nlhs > 2)
		return "Only returns two values!";

	size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims > 5)
		return "Image can have a maximum of five dimensions!";

	return "";
}

void MexMinMax::usage(std::vector<std::string>& outArgs, std::vector<std::string>& inArgs) const
{
	inArgs.push_back("arrayIn");
	inArgs.push_back("[device]");
	outArgs.push_back("minOut");
	outArgs.push_back("maxOut");
}

void MexMinMax::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("This returns the global min and max values.");

	helpLines.push_back("\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.");
	helpLines.push_back("\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine");
	helpLines.push_back("\t\thow to stride or jump to the next spatial block.");
	helpLines.push_back("");

	helpLines.push_back("\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.");
	helpLines.push_back("\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split");
	helpLines.push_back("\t\tthe data across multiple devices.");
	helpLines.push_back("");

	helpLines.push_back("\tminOut = This is the minimum value found in the input.");
	helpLines.push_back("\tmaxOut = This is the maximum value found in the input.");
}

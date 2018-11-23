#include "MexCommand.h"
#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"
#include "MexKernel.h"

void MexGetMinMax::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	// TODO: Why is this the only device with a 0 default?
	int device = 0;

	if (!mxIsEmpty(prhs[1]))
		device = mat_to_c((int)mxGetScalar(prhs[1]));

	ImageDimensions imageDims;
	if (mxIsLogical(prhs[0]))
	{
		bool* imageInPtr, minVal, maxVal;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims);

		getMinMax(imageInPtr, imageDims.getNumElements(), minVal, maxVal, device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageInPtr, minVal, maxVal;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims);

		getMinMax(imageInPtr, imageDims.getNumElements(), minVal, maxVal, device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageInPtr, minVal, maxVal;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims);

		getMinMax(imageInPtr, imageDims.getNumElements(), minVal, maxVal, device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageInPtr, minVal, maxVal;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims);

		getMinMax(imageInPtr, imageDims.getNumElements(), minVal, maxVal, device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageInPtr, minVal, maxVal;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims);

		getMinMax(imageInPtr, imageDims.getNumElements(), minVal, maxVal, device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageInPtr, minVal, maxVal;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims);

		getMinMax(imageInPtr, imageDims.getNumElements(), minVal, maxVal, device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageInPtr, minVal, maxVal;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims);

		getMinMax(imageInPtr, imageDims.getNumElements(), minVal, maxVal, device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageInPtr, minVal, maxVal;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims);

		getMinMax(imageInPtr, imageDims.getNumElements(), minVal, maxVal, device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexGetMinMax::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	if (nrhs < 1 || nrhs > 2)
		return "Incorrect number of inputs!";

	if (nlhs != 2)
		return "Requires two outputs!";

	std::size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims > 5)
		return "Image can have a maximum of five dimensions!";

	return "";
}

void MexGetMinMax::usage(std::vector<std::string>& outArgs, std::vector<std::string>& inArgs) const
{
	inArgs.push_back("arrayIn");
	inArgs.push_back("[device]");
	outArgs.push_back("minValue");
	outArgs.push_back("maxValue");
}

void MexGetMinMax::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("This function finds the lowest and highest value in the array that is passed in.");

	helpLines.push_back("\timageIn = This is a one to five dimensional array.");
	helpLines.push_back("");

	helpLines.push_back("\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.");
	helpLines.push_back("\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split");
	helpLines.push_back("\t\tthe data across multiple devices.");
	helpLines.push_back("");

	helpLines.push_back("\tminValue = This is the lowest value found in the array.");
	helpLines.push_back("\tmaxValue = This is the highest value found in the array.");
}

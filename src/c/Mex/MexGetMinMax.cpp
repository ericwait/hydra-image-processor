#include "MexCommand.h"
#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"
#include "MexKernel.h"

template <typename T>
void MexMinMax_run(const mxArray* inIm, mxArray** outMin, mxArray** outMax, int device)
{
	T* imageInPtr;
	T minVal;
	T maxVal;

	ImageDimensions imageDims;
	Script::setupInputPointers(inIm, imageDims, &imageInPtr);

	getMinMax(imageInPtr, imageDims.getNumElements(), minVal, maxVal, device);

	*outMin = mxCreateDoubleScalar(minVal);
	*outMax = mxCreateDoubleScalar(maxVal);
}

void MexGetMinMax::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	// TODO: Why is this the only device with a 0 default?
	int device = 0;

	if (!mxIsEmpty(prhs[1]))
		device = mat_to_c((int)mxGetScalar(prhs[1]));

	ImageDimensions imageDims;
	if (mxIsLogical(prhs[0]))
	{
		MexMinMax_run<bool>(prhs[0], &plhs[0], &plhs[1], device);
	}
	else if (mxIsUint8(prhs[0]))
	{
		MexMinMax_run<uint8_t>(prhs[0], &plhs[0], &plhs[1], device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		MexMinMax_run<uint16_t>(prhs[0], &plhs[0], &plhs[1], device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		MexMinMax_run<int16_t>(prhs[0], &plhs[0], &plhs[1], device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		MexMinMax_run<uint32_t>(prhs[0], &plhs[0], &plhs[1], device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		MexMinMax_run<int32_t>(prhs[0], &plhs[0], &plhs[1], device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		MexMinMax_run<float>(prhs[0], &plhs[0], &plhs[1], device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		MexMinMax_run<double>(prhs[0], &plhs[0], &plhs[1], device);
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

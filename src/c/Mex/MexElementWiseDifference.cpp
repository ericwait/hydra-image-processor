#include "MexCommand.h"
#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"
#include "MexKernel.h"

template <typename T>
void MexDiff_run(const mxArray* inIm1, const mxArray* inIm2, mxArray** outIm, int device)
{
	Script::DimInfo inInfo1 = Script::getDimInfo(inIm1);
	Script::DimInfo inInfo2 = Script::getDimInfo(inIm1);
	Script::DimInfo outInfo = Script::maxDims(inInfo1, inInfo2);

	ImageView<T> image1In = Script::wrapInputImage<T>(inIm1, inInfo1);
	ImageView<T> image2In = Script::wrapInputImage<T>(inIm2, inInfo2);
	ImageView<T> imageOut = Script::createOutputImage<T>(outIm, outInfo);

	elementWiseDifference(image1In, image2In, imageOut, device);
}

void MexElementWiseDifference::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	int device = -1;

	if (!mxIsEmpty(prhs[2]))
		device = mat_to_c((int)mxGetScalar(prhs[3]));

	ImageDimensions imageDims;
	if (mxIsLogical(prhs[0]))
	{
		MexDiff_run<bool>(prhs[0], prhs[1], &plhs[0], device);
	}
	else if (mxIsUint8(prhs[0]))
	{
		MexDiff_run<uint8_t>(prhs[0], prhs[1], &plhs[0], device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		MexDiff_run<uint16_t>(prhs[0], prhs[1], &plhs[0], device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		MexDiff_run<int16_t>(prhs[0], prhs[1], &plhs[0], device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		MexDiff_run<uint32_t>(prhs[0], prhs[1], &plhs[0], device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		MexDiff_run<int32_t>(prhs[0], prhs[1], &plhs[0], device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		MexDiff_run<float>(prhs[0], prhs[1], &plhs[0], device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		MexDiff_run<double>(prhs[0], prhs[1], &plhs[0], device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexElementWiseDifference::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	if (nrhs != 3)
		return "Incorrect number of inputs!";

	if (nlhs != 1)
		return "Requires one output!";

	std::size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims > 5)
		return "Image1 can have a maximum of five dimensions!";

	imgNumDims = mxGetNumberOfDimensions(prhs[1]);
	if (imgNumDims > 5)
		return "Image2 can have a maximum of five dimensions!";

	return "";
}

void MexElementWiseDifference::usage(std::vector<std::string>& outArgs, std::vector<std::string>& inArgs) const
{
	inArgs.push_back("array1In");
	inArgs.push_back("array2In");
	inArgs.push_back("[device]");
	outArgs.push_back("arrayOut");
}

void MexElementWiseDifference::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("This subtracts the second array from the first, element by element (A-B).");

	helpLines.push_back("\timage1In = This is a one to five dimensional array. The first three dimensions are treated as spatial.");
	helpLines.push_back("\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine");
	helpLines.push_back("\t\thow to stride or jump to the next spatial block.");
	helpLines.push_back("");

	helpLines.push_back("\timage2In = This is a one to five dimensional array. The first three dimensions are treated as spatial.");
	helpLines.push_back("\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine");
	helpLines.push_back("\t\thow to stride or jump to the next spatial block.");
	helpLines.push_back("");

	helpLines.push_back("\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.");
	helpLines.push_back("\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split");
	helpLines.push_back("\t\tthe data across multiple devices.");
	helpLines.push_back("");

	helpLines.push_back("\timageOut = This will be an array of the same type and shape as the input array.");
}

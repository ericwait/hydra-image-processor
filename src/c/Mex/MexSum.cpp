#include "MexCommand.h"
#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"
#include "MexKernel.h"

void MexSum::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	int device = -1;

	if (!mxIsEmpty(prhs[1]))
		device = mat_to_c((int)mxGetScalar(prhs[1]));

	ImageDimensions imageDims;
	double outVal = 0.0;
	if (mxIsLogical(prhs[0]))
	{
		bool* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<bool> imageIn(imageInPtr, imageDims);

		size_t out = 0;
		sum(imageIn, out, device);
		outVal = double(out);
	}
	else if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<unsigned char> imageIn(imageInPtr, imageDims);

		size_t out = 0;
		sum(imageIn, out, device);
		outVal = double(out);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<unsigned short> imageIn(imageInPtr, imageDims);

		size_t out = 0;
		sum(imageIn, out, device);
		outVal = double(out);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<short> imageIn(imageInPtr, imageDims);

		long long out = 0;
		sum(imageIn, out, device);
		outVal = double(out);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<unsigned int> imageIn(imageInPtr, imageDims);

		size_t out = 0;
		sum(imageIn, out, device);
		outVal = double(out);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<int> imageIn(imageInPtr, imageDims);

		long long out = 0;
		sum(imageIn, out, device);
		outVal = double(out);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<float> imageIn(imageInPtr, imageDims);

		double out = 0;
		sum(imageIn, out, device);
		outVal = double(out);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageInPtr;
		setupImagePointers(prhs[0], &imageInPtr, imageDims);

		ImageContainer<double> imageIn(imageInPtr, imageDims);

		double out = 0;
		sum(imageIn, out, device);
		outVal = double(out);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}

	plhs[0] = mxCreateDoubleScalar(outVal);
}

std::string MexSum::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	if (nrhs != 2)
		return "Incorrect number of inputs!";

	if (nlhs != 1)
		return "Requires one output!";

	size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims > 5)
		return "Image can have a maximum of five dimensions!";

	return "";
}

void MexSum::usage(std::vector<std::string>& outArgs, std::vector<std::string>& inArgs) const
{
	inArgs.push_back("arrayIn");
	inArgs.push_back("[device]");
	outArgs.push_back("valueOut");
}

void MexSum::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("This sums up the entire array in.");

	helpLines.push_back("\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.");
	helpLines.push_back("\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine");
	helpLines.push_back("\t\thow to stride or jump to the next spatial block.");
	helpLines.push_back("");

	helpLines.push_back("\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.");
	helpLines.push_back("\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split");
	helpLines.push_back("\t\tthe data across multiple devices.");
	helpLines.push_back("");

	helpLines.push_back("\tvalueOut = This is the summation of the entire array.");
}

#include "MexCommand.h"
#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"
#include "MexKernel.h"

template <typename T>
void MexMultiplySum_run(const mxArray* inIm, mxArray** outIm, ImageContainer<float> kernel, int numIterations, int device)
{
	T* imageInPtr;
	T* imageOutPtr;

	ImageDimensions imageDims;
	Script::setupImagePointers(inIm, &imageInPtr, imageDims, outIm, &imageOutPtr);

	ImageContainer<T> imageIn(imageInPtr, imageDims);
	ImageContainer<T> imageOut(imageOutPtr, imageDims);

	multiplySum(imageIn, imageOut, kernel, numIterations, device);
}


void MexMultiplySum::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	int device = -1;
	int numIterations = 1;

	if (!mxIsEmpty(prhs[2]))
		numIterations = int(mxGetScalar(prhs[2]));

	if (!mxIsEmpty(prhs[3]))
		device = mat_to_c((int)mxGetScalar(prhs[3]));

	ImageContainer<float> kernel = getKernel(prhs[1]);

	if (kernel.getDims().getNumElements() == 0)
	{
		kernel.clear();
		return;
	}

	if (mxIsLogical(prhs[0]))
	{
		MexMultiplySum_run<bool>(prhs[0], &plhs[0], kernel, numIterations, device);
	}
	else if (mxIsUint8(prhs[0]))
	{
		MexMultiplySum_run<uint8_t>(prhs[0], &plhs[0], kernel, numIterations, device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		MexMultiplySum_run<uint16_t>(prhs[0], &plhs[0], kernel, numIterations, device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		MexMultiplySum_run<int16_t>(prhs[0], &plhs[0], kernel, numIterations, device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		MexMultiplySum_run<uint32_t>(prhs[0], &plhs[0], kernel, numIterations, device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		MexMultiplySum_run<int32_t>(prhs[0], &plhs[0], kernel, numIterations, device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		MexMultiplySum_run<float>(prhs[0], &plhs[0], kernel, numIterations, device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		MexMultiplySum_run<double>(prhs[0], &plhs[0], kernel, numIterations, device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}

	kernel.clear();
}

std::string MexMultiplySum::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	if (nrhs != 4)
		return "Incorrect number of inputs!";

	if (nlhs != 1)
		return "Requires one output!";

	std::size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims > 5)
		return "Image can have a maximum of five dimensions!";

	std::size_t kernDims = mxGetNumberOfDimensions(prhs[1]);
	if (kernDims < 1 || kernDims>3)
		return "Kernel can only be either 1-D, 2-D, or 3-D!";

	if (!mxIsEmpty(prhs[2]))
	{
		int numIter = (int)mxGetScalar(prhs[2]);
		if (numIter < 1)
			return "Number of iterations must be 1 or greater!";
	}


	return "";
}

void MexMultiplySum::usage(std::vector<std::string>& outArgs, std::vector<std::string>& inArgs) const
{
	inArgs.push_back("arrayIn");
	inArgs.push_back("kernel");
	inArgs.push_back("[numIterations]");
	inArgs.push_back("[device]");
	outArgs.push_back("arrayOut");
}

void MexMultiplySum::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("Multiplies the kernel with the neighboring values and sums these new values.");

	helpLines.push_back("\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.");
	helpLines.push_back("\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine");
	helpLines.push_back("\t\thow to stride or jump to the next spatial block.");
	helpLines.push_back("");

	helpLines.push_back("\tkernel = This is a one to three dimensional array that will be used to determine neighborhood operations.");
	helpLines.push_back("\t\tIn this case, the positions in the kernel that do not equal zeros will be evaluated.");
	helpLines.push_back("\t\tIn other words, this can be viewed as a structuring element for the max neighborhood.");
	helpLines.push_back("");

	helpLines.push_back("\tnumIterations (optional) =  This is the number of iterations to run the max filter for a given position.");
	helpLines.push_back("\t\tThis is useful for growing regions by the shape of the structuring element or for very large neighborhoods.");
	helpLines.push_back("\t\tCan be empty an array [].");
	helpLines.push_back("");

	helpLines.push_back("\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.");
	helpLines.push_back("\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split");
	helpLines.push_back("\t\tthe data across multiple devices.");
	helpLines.push_back("");

	helpLines.push_back("\timageOut = This will be an array of the same type and shape as the input array.");
}

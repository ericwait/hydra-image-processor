#include "MexCommand.h"
#include "../Cuda/Vec.h"
#include "../Cuda/CWrappers.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageContainer.h"
#include "MexKernel.h"

void MexIdentityFilter::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	int device = -1;

	ImageDimensions imageDims;
	if ( mxIsLogical(prhs[0]) )
	{
		bool* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims, &plhs[0], &imageOutPtr);

		ImageContainer<bool> imageIn(imageInPtr, imageDims);
		ImageContainer<bool> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);

	}
	else if ( mxIsUint8(prhs[0]) )
	{
		unsigned char* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims, &plhs[0], &imageOutPtr);

		ImageContainer<unsigned char> imageIn(imageInPtr, imageDims);
		ImageContainer<unsigned char> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( mxIsUint16(prhs[0]) )
	{
		unsigned short* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims, &plhs[0], &imageOutPtr);

		ImageContainer<unsigned short> imageIn(imageInPtr, imageDims);
		ImageContainer<unsigned short> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( mxIsInt16(prhs[0]) )
	{
		short* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims, &plhs[0], &imageOutPtr);

		ImageContainer<short> imageIn(imageInPtr, imageDims);
		ImageContainer<short> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( mxIsUint32(prhs[0]) )
	{
		unsigned int* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims, &plhs[0], &imageOutPtr);

		ImageContainer<unsigned int> imageIn(imageInPtr, imageDims);
		ImageContainer<unsigned int> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( mxIsInt32(prhs[0]) )
	{
		int* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims, &plhs[0], &imageOutPtr);

		ImageContainer<int> imageIn(imageInPtr, imageDims);
		ImageContainer<int> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( mxIsSingle(prhs[0]) )
	{
		float* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims, &plhs[0], &imageOutPtr);

		ImageContainer<float> imageIn(imageInPtr, imageDims);
		ImageContainer<float> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else if ( mxIsDouble(prhs[0]) )
	{
		double* imageInPtr, *imageOutPtr;
		Script::setupImagePointers(prhs[0], &imageInPtr, imageDims, &plhs[0], &imageOutPtr);

		ImageContainer<double> imageIn(imageInPtr, imageDims);
		ImageContainer<double> imageOut(imageOutPtr, imageDims);

		identityFilter(imageIn, imageOut, device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexIdentityFilter::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
	if ( nrhs != 2 )
		return "Incorrect number of inputs!";

	if ( nlhs != 1 )
		return "Requires one output!";

	std::size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if ( imgNumDims > 5 )
		return "Image can have a maximum of five dimensions!";


	return "";
}

void MexIdentityFilter::usage(std::vector<std::string>& outArgs, std::vector<std::string>& inArgs) const
{
	inArgs.push_back("arrayIn");
	inArgs.push_back("[device]");
	outArgs.push_back("arrayOut");
}

void MexIdentityFilter::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("Identity Filter for testing. Copies image data to GPU memory and back into output image");

	helpLines.push_back("\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.");
	helpLines.push_back("\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine");
	helpLines.push_back("\t\thow to stride or jump to the next spatial block.");
	helpLines.push_back("");

	helpLines.push_back("\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.");
	helpLines.push_back("\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split");
	helpLines.push_back("\t\tthe data across multiple devices.");
	helpLines.push_back("");

	helpLines.push_back("\timageOut = This will be an array of the same type and shape as the input array.");
}

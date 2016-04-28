#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexContrastEnhancement::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	int device = 0;

	if (nrhs>3)
		device = mat_to_c((int)mxGetScalar(prhs[3]));

	double* sigmasD = (double*)mxGetData(prhs[1]);
	double* neighborhoodD = (double*)mxGetData(prhs[2]);


	Vec<float> sigmas((float)sigmasD[0],(float)sigmasD[1],(float)sigmasD[2]);
	Vec<size_t> neighborhood((size_t)neighborhoodD[0],(size_t)neighborhoodD[1],(size_t)neighborhoodD[2]);
	
	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		contrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		contrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		contrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		contrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		contrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		contrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		contrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexContrastEnhancement::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	if (nrhs<3 || nrhs>4)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3)
		return "Image can have a maximum of three dimensions!";

	size_t numEl= mxGetNumberOfElements(prhs[1]);
	if (numEl!=3 || !mxIsDouble(prhs[1]))
		return "Sigmas has to be an array of three doubles!";

	numEl = mxGetNumberOfElements(prhs[2]);
	if (numEl!=3 || !mxIsDouble(prhs[2]))
		return "Median neighborhood has to be an array of three doubles!";

	return "";
}

void MexContrastEnhancement::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
{
	inArgs.push_back("imageIn");
	inArgs.push_back("sigma");
	inArgs.push_back("MedianNeighborhood");
	inArgs.push_back("device");

	outArgs.push_back("imageOut");
}

void MexContrastEnhancement::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("This attempts to increase contrast by removing noise as proposed by Michel et al. This starts with subtracting off a highly smoothed version of imageIn followed by median filter.");
	
	helpLines.push_back("\tImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).");
	helpLines.push_back("\tSigma -- these values will create a n-dimensional Gaussian kernel to get a smoothed image that will be subtracted of the original.");
	helpLines.push_back("\t\tN is the number of dimensions of imageIn");
	helpLines.push_back("\t\tThe larger the sigma the more object preserving the high pass filter will be (e.g. sigma > 35)");
	helpLines.push_back("\tMedianNeighborhood -- this is the neighborhood size in each dimension that will be evaluated for the median neighborhood filter.");
	helpLines.push_back("\tDevice -- this is an optional parameter that indicates which Cuda capable device to use.");

	helpLines.push_back("\tImageOut -- will have the same dimensions and type as imageIn.");
}
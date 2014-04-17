#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.cuh"

void MexContrastEnhancement::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>3)
		device = mat_to_c((int)mxGetScalar(prhs[3]));

	double* sigmasD = (double*)mxGetData(prhs[1]);
	double* neighborhoodD = (double*)mxGetData(prhs[2]);

	Vec<float> sigmas((float)sigmasD[0],(float)sigmasD[1],(float)sigmasD[2]);
	Vec<size_t> neighborhood((int)neighborhoodD[0],(int)neighborhoodD[1],(int)neighborhoodD[2]);
	
	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		cContrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		cContrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		cContrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		cContrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		cContrastEnhancement(imageIn,imageDims,sigmas,neighborhood,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexContrastEnhancement::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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

std::string MexContrastEnhancement::printUsage()
{
	return "imageOut = CudaMex('ContrastEnhancement',imageIn,[sigmaX,sigmaY,sigmaZ],[MedianNeighborhoodX,MedianNeighborhoodY,MedianNeighborhoodZ],[device]);";
}

std::string MexContrastEnhancement::printHelp()
{
	std::string msg = "\tContrastEnancement will do a high-pass background subtraction followed by a median smoothing.\n";
	msg += "\tsigmaX, sigmaY, and sigmaZ correspond to the Gaussian smoothing kernel in those dimensions.\n";
	msg += "\tMedianNeighborhoodX, MedianNeighborhoodY, and MedianNeighborhoodZ relate to how big the median smoothing window will be.\n";
	msg += "\n";
	return msg;
}
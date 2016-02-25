#include "MexCommand.h"

#include "MexCommand.h"
#include "CWrappers.h"
#include "Vec.h"

void MexMarkovRandomFieldDenoiser::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	double maxIterations = mxGetScalar(prhs[1]);

	Vec<size_t> imageDims;
// 	if (mxIsUint8(prhs[0]))
// 	{
// 		unsigned char* imageIn,* imageOut;
// 		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
// 
// 		markovRandomFieldDenoiser(imageIn,imageDims,maxIterations,&imageOut,device);
// 	}
// 	else if (mxIsUint16(prhs[0]))
// 	{
// 		unsigned short* imageIn,* imageOut;
// 		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
// 
// 		markovRandomFieldDenoiser(imageIn,imageDims,maxIterations,&imageOut,device);
// 	}
// 	else if (mxIsInt16(prhs[0]))
// 	{
// 		short* imageIn,* imageOut;
// 		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
// 
// 		markovRandomFieldDenoiser(imageIn,imageDims,maxIterations,&imageOut,device);
// 	}
// 	else if (mxIsUint32(prhs[0]))
// 	{
// 		unsigned int* imageIn,* imageOut;
// 		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
// 
// 		markovRandomFieldDenoiser(imageIn,imageDims,maxIterations,&imageOut,device);
// 	}
// 	else if (mxIsInt32(prhs[0]))
// 	{
// 		int* imageIn,* imageOut;
// 		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
// 
// 		markovRandomFieldDenoiser(imageIn,imageDims,maxIterations,&imageOut,device);
// 	}
//	else 
	if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		markovRandomFieldDenoiser(imageIn,imageDims,int(maxIterations),&imageOut,device);
	}
// 	else if (mxIsDouble(prhs[0]))
// 	{
// 		double* imageIn,* imageOut;
// 		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
// 
// 		markovRandomFieldDenoiser(imageIn,imageDims,maxIterations,&imageOut,device);
// 	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexMarkovRandomFieldDenoiser::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs<2 || nrhs>3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3)
		return "Image can have a maximum of three dimensions!";

	return "";
}

std::string MexMarkovRandomFieldDenoiser::printUsage()
{
	return "imageOut = CudaMex('MarkovRandomFieldDenoiser',imageIn,maxIterations,[device]);";
}

std::string MexMarkovRandomFieldDenoiser::printHelp()
{
	std::string msg = "\tMarkov Random Field Denoiser will denoise the image using a noise estimation iteratively until the image\n";
	msg += "\tmatches the noise model or the max iterations is reached.\n";
	msg += "\tSee Ceccarelli, M. (2007). \"A Finite Markov Random Field approach to fast edge-preserving image recovery.\"\n";
	msg += "Image and Vision Computing 25(6): 792-804.\n";
	msg += "\n";
	return msg;
}
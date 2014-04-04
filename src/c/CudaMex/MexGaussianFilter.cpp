#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"

 void MexGaussianFilter::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>3)
		 device = mat_to_c((int)mxGetScalar(prhs[3]));

	 Vec<size_t> imageDims;
	 HostPixelType* imageIn, * imageOut;
	 CudaProcessBuffer cudaBuffer(device);
	 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

	 double* sigmasMat = (double*)mxGetData(prhs[1]);
	 Vec<float> sigmas((float)(sigmasMat[0]),(float)(sigmasMat[1]),(float)(sigmasMat[2]));

	 cudaBuffer.gaussianFilter(imageIn,imageDims,sigmas,&imageOut);
 }
 
 std::string MexGaussianFilter::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
 	if (nrhs<2 || nrhs>3)
 		return "Incorrect number of inputs!";
 
 	if (nlhs!=1)
 		return "Requires one output!";
 
 	if (!mxIsUint8(prhs[0]))
 		return "Image has to be formated as a uint8!";
 
 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
 	if (numDims>3 || numDims<2)
 		return "Image can only be either 2D or 3D!";
 
 	size_t numEl = mxGetNumberOfElements(prhs[1]);
 	if (numEl!=3 || !mxIsDouble(prhs[1]))
 		return "Sigmas must be an array of three doubles!";
 
 	return "";
 }
 
 std::string MexGaussianFilter::printUsage()
 {
	 return "imageOut = CudaMex('GaussianFilter',imageIn,[sigmaX,sigmaY,sigmaZ],[device]);";

 }

 std::string MexGaussianFilter::printHelp()
 {
	 std::string msg = "\tsigmaX, sigmaY, and sigmaZ are the parameters to set up the Gaussian smoothing kernel.\n";
	 msg += "\n";
	 return msg;
 }
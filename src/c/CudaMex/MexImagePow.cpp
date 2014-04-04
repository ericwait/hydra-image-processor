#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"
 
 void MexImagePow::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>2)
		 device = mat_to_c((int)mxGetScalar(prhs[2]));

	 Vec<size_t> imageDims;
	 HostPixelType* imageIn, * imageOut;
	 CudaProcessBuffer cudaBuffer(device);
	 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
 
 	double p = mxGetScalar(prhs[1]);
 	cudaBuffer.imagePow(imageIn,imageDims,p,&imageOut);
 }
 
 std::string MexImagePow::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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
 
 	if (!mxIsDouble(prhs[1]))
 		return "Power has to be a single double!";
 
 	return "";
 }
 
 std::string MexImagePow::printUsage()
 {
 	return "imageOut = CudaMex('ImagePow',imageIn,power,[device]);";
 }

 std::string MexImagePow::printHelp()
 {
	 std::string msg = "\tPower must be a double and will be ceilinged if input is integer.\n";
	 msg += "\tImageOut will not roll over.  Values are clamped to the range of the image space.\n";
	 msg += "\tImageOut will have the same dimensions as imageIn.\n";
	 msg += "\n";
	 return msg;
 }
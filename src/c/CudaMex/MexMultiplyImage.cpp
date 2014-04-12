#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"
 
 void MexMultiplyImage::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>2)
		 device = mat_to_c((int)mxGetScalar(prhs[2]));

	 Vec<size_t> imageDims;
	 HostPixelType* imageIn, * imageOut;
	 CudaProcessBuffer cudaBuffer(device);
	 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
 
 	double multiplier = mxGetScalar(prhs[1]);
 
 	cudaBuffer.multiplyImage(imageIn,imageDims,multiplier,&imageOut);
 }
 
 std::string MexMultiplyImage::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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
 		return "Multiplier needs to be a single double!";
 
 	return "";
 }
 
 std::string MexMultiplyImage::printUsage()
 {
 	return "imageOut = CudaMex('MultiplyImage',imageIn,multiplier,[device]);";
 }

 std::string MexMultiplyImage::printHelp()
 {
	 std::string msg = "\tMultiplier must be a double.\n";
	 msg += "\tImageOut will not roll over.  Values are clamped to the range of the image space.\n";
	 msg += "\tImageOut will have the same dimensions as imageIn.\n";
	 msg += "\n";
	 return msg;
 }
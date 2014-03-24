#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"
 
 void AddConstant::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
 {
	 Vec<size_t> imageDims;
	 HostPixelType* imageIn, * imageOut;
	 CudaProcessBuffer cudaBuffer;
	 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
 
 	double additive = mxGetScalar(prhs[1]);
 
 	cudaBuffer.addConstant(imageIn,imageDims,additive,&imageOut);
 }
 
 std::string AddConstant::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
 {
 	if (nrhs!=2)
 		return "Incorrect number of inputs!";
 
 	if (nlhs!=1)
 		return "Requires one output!";
 
 	if (!mxIsUint8(prhs[0]))
 		return "Image has to be formated as a uint8!";
 
 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
 	if (numDims>3 || numDims<2)
 		return "Image can only be either 2D or 3D!";
 
 	if (!mxIsDouble(prhs[1]))
 		return "Additive needs to be a single double!";
 
 	return "";
 }
 
 std::string AddConstant::printUsage()
 {
	 return "imageOut = CudaMex('AddConstant',imageIn,additive);";
 }

 std::string AddConstant::printHelp()
 {
	 std::string msg = "\tAdditive must be a double and will be floored if input is integer.\n";
	 msg += "\tImageOut will not roll over.  Values are clamped to the range of the image space.\n";
	 msg += "\tImageOut will have the same dimensions as imageIn.\n";
	 msg += "\n";
	 return msg;
 }
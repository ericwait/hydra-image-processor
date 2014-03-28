#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"
 
 void AddImageWith::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>3)
		 device = mat_to_c((int)mxGetScalar(prhs[3]));

	 Vec<size_t> imageDims1, imageDims2;
	 HostPixelType* imageIn1, * imageIn2, * imageOut;
	 CudaProcessBuffer cudaBuffer(device);
	 setupImagePointers(prhs[0],&imageIn1,&imageDims1,&plhs[0],&imageOut);
	 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

	 if (imageDims1!=imageDims2)
		 mexErrMsgTxt("Image dimensions must agree!");

	 double additive = mxGetScalar(prhs[2]);

	 cudaBuffer.addImageWith(imageIn1,imageIn2,imageDims1,additive,&imageOut);
 }
 
 std::string AddImageWith::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
 	if (nrhs<3 || nrhs>4)
 		return "Incorrect number of inputs!";
 
 	if (nlhs!=1)
 		return "Requires one output!";
 
 	if (!mxIsUint8(prhs[0]) || !mxIsUint8(prhs[1]))
 		return "Images has to be formated as a uint8!";
 
 	size_t numDims1 = mxGetNumberOfDimensions(prhs[0]);
 	if (numDims1>3 || numDims1<2)
 		return "Image can only be either 2D or 3D!";
 
 	size_t numDims2 = mxGetNumberOfDimensions(prhs[1]);
 	if (numDims2>3 || numDims2<2)
 		return "Image can only be either 2D or 3D!";

	if (numDims1!=numDims2)
		return "Image dimensions must agree!";
 
 	if (!mxIsDouble(prhs[2]))
 		return "Factor needs to be a double!";
 
 	return "";
 }
 
 std::string AddImageWith::printUsage()
 {
	 return "imageOut = CudaMex('AddImageWith',imageIn1,imageIn2,factor,[device]);";
 }

 std::string AddImageWith::printHelp()
 {
	 std::string msg = "\tWhere factor is a multiplier on imageIn2.  Pixel = imageIn1 + factor*imageIn2.";
	 msg += "\tfactor is used in place of a SubImageWith (e.g. factor = -1).\n";
	 msg += "\tPixel value is floored at assignment only when integer.\n";
	 msg += "\tImageIn1 and ImageIn2 must have the same dimensions.\n";
	 msg += "\tImageOut will have the same dimensions as the input images.\n";
	 msg += "\n";
	 return msg;
 }
#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.cuh"
 
 void MexMultiplyImage::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>2)
		 device = mat_to_c((int)mxGetScalar(prhs[2]));
 
 	double multiplier = mxGetScalar(prhs[1]);
 
	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		cMultiplyImage(imageIn,imageDims,multiplier,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		cMultiplyImage(imageIn,imageDims,multiplier,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		cMultiplyImage(imageIn,imageDims,multiplier,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		cMultiplyImage(imageIn,imageDims,multiplier,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		cMultiplyImage(imageIn,imageDims,multiplier,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
 }
 
 std::string MexMultiplyImage::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
 	if (nrhs<2 || nrhs>3)
 		return "Incorrect number of inputs!";
 
 	if (nlhs!=1)
 		return "Requires one output!";
 
 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3)
		return "Image can have a maximum of three dimensions!";
 
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
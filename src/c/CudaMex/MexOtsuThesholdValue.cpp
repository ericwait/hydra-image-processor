#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.cuh"
 
 void MexOtsuThesholdValue::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>1)
		 device = mat_to_c((int)mxGetScalar(prhs[1]));
 
 	double thresh = 0.0;

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = cOtsuThresholdValue(imageIn,imageDims,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = cOtsuThresholdValue(imageIn,imageDims,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = cOtsuThresholdValue(imageIn,imageDims,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = cOtsuThresholdValue(imageIn,imageDims,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = cOtsuThresholdValue(imageIn,imageDims,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
 
 	plhs[0] = mxCreateDoubleScalar(thresh);
 }
 
 std::string MexOtsuThesholdValue::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
 	if (nrhs<1 || nrhs>2)
 		return "Incorrect number of inputs!";
 
 	if (nlhs!=1)
 		return "Requires one output!";
 
 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
 	if (numDims>3)
 		return "Image can have a maximum of three dimensions!";
 
 	return "";
 }
 
 std::string MexOtsuThesholdValue::printUsage()
 {
 	return "threshold = CudaMex('OtsuThesholdValue',imageIn,[device]);";
 }

 std::string MexOtsuThesholdValue::printHelp()
 {
	 std::string msg = "\tCalculates the optimal two class threshold using Otsu's method.\n";
	 msg += "\n";
	 return msg;
 }
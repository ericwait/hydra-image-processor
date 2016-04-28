#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

 void MexOtsuThresholdValue::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
 {
	 int device = 0;

	 if (nrhs>1)
		 device = mat_to_c((int)mxGetScalar(prhs[1]));

 	double thresh = 0.0;

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = otsuThresholdValue(imageIn,imageDims,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = otsuThresholdValue(imageIn,imageDims,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = otsuThresholdValue(imageIn,imageDims,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = otsuThresholdValue(imageIn,imageDims,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = otsuThresholdValue(imageIn,imageDims,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = otsuThresholdValue(imageIn,imageDims,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		thresh = otsuThresholdValue(imageIn,imageDims,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}

 	plhs[0] = mxCreateDoubleScalar(thresh);
 }

 std::string MexOtsuThresholdValue::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
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

 void MexOtsuThresholdValue::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
 {
 	inArgs.push_back("imageIn");

inArgs.push_back("device");
outArgs.push_back("threshold");
 }

 void MexOtsuThresholdValue::help(std::vector<std::string>& helpLines) const
 {
//\	 std::string msg = "\tCalculates the optimal two class threshold using Otsu's method.\n";
//\	 msg += "\n";
//\	 return msg;
 }

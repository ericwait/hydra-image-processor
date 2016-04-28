#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexMinMax::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	int device = 0;

	if (nrhs>1)
		device = mat_to_c((int)mxGetScalar(prhs[1]));

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned char minVal, maxVal;
		getMinMax(imageIn,imageDims,minVal,maxVal,device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned short minVal, maxVal;
		getMinMax(imageIn,imageDims,minVal,maxVal,device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		short minVal, maxVal;

		getMinMax(imageIn,imageDims,minVal,maxVal,device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned int minVal, maxVal;
		getMinMax(imageIn,imageDims,minVal,maxVal,device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		int minVal, maxVal;

		getMinMax(imageIn,imageDims,minVal,maxVal,device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		float minVal, maxVal;
		getMinMax(imageIn,imageDims,minVal,maxVal,device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		double minVal, maxVal;
		getMinMax(imageIn,imageDims,minVal,maxVal,device);
		plhs[0] = mxCreateDoubleScalar(minVal);
		plhs[1] = mxCreateDoubleScalar(maxVal);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexMinMax::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	if (nrhs<1 || nrhs>2)
		return "Incorrect number of inputs!";

	if (nlhs!=2)
		return "Requires two outputs!";

	 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	 	if (numDims>3)
	 		return "Image can have a maximum of three dimensions!";

	return "";
}

void MexMinMax::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
{
	inArgs.push_back("imageIn");
inArgs.push_back("device");

outArgs.push_back("min");
outArgs.push_back("max");
}

void MexMinMax::help(std::vector<std::string>& helpLines) const
{
//\	std::string msg = "\tReturns the minimum and maximum values.\n";
//\	msg += "\n";
//\	return msg;
}

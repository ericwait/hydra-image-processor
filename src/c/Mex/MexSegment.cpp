#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"
#include "CHelpers.h"

void MexSegment::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	int device = 0;

	if (nrhs>3)
		device = mat_to_c((int)mxGetScalar(prhs[3]));

	double alpha = mxGetScalar(prhs[1]);
	double* radiusMex = (double*)mxGetData(prhs[2]);
	Vec<size_t> radius((size_t)radiusMex[0],(size_t)radiusMex[1],(size_t)radiusMex[2]);

	Vec<size_t> kernDims;
	float* kern = createEllipsoidKernel(radius,kernDims);

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		segment(imageIn,imageDims,alpha,kernDims,kern,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		segment(imageIn,imageDims,alpha,kernDims,kern,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		segment(imageIn,imageDims,alpha,kernDims,kern,&imageOut,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		segment(imageIn,imageDims,alpha,kernDims,kern,&imageOut,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		segment(imageIn,imageDims,alpha,kernDims,kern,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		segment(imageIn,imageDims,alpha,kernDims,kern,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		segment(imageIn,imageDims,alpha,kernDims,kern,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexSegment::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	if (nrhs<3 || nrhs>4)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3)
		return "Image can have a maximum of three dimensions!";

	if (!mxIsDouble(prhs[1]))
		return "Alpha has to be a single double!";

	size_t numEl = mxGetNumberOfElements(prhs[2]);
	if (numEl!=3 || !mxIsDouble(prhs[2]))
		return "Median neighborhood has to be an array of three doubles!";

	return "";
}

void MexSegment::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
{
	inArgs.push_back("imageIn");
	inArgs.push_back("alpha");
	inArgs.push_back("MorphClosure");
	inArgs.push_back("device");
	outArgs.push_back("imageOut");
}

void MexSegment::help(std::vector<std::string>& helpLines) const
{
//\	std::string msg = "\tSegmentaion is done by applying an Otsu adaptive threshold (which can be modified by the alpha multiplier).\n";
//\	msg += "\tA morphological closing is then applied using a ellipsoid neighborhood with the MorphClosure dimensions.\n";
//\	msg += "\n";
//\	return msg;
}

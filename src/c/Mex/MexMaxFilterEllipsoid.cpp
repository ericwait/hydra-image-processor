#include "MexCommand.h"
#include "CHelpers.h"
#include "Vec.h"
#include "CWrappers.h"

void MexMaxFilterEllipsoid::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	double* radiiD = (double*)mxGetData(prhs[1]);

	Vec<size_t> radii((size_t)radiiD[0],(size_t)radiiD[1],(size_t)radiiD[2]);
	Vec<size_t> kernDims;
	float* circleKernel = createEllipsoidKernel(radii,kernDims);

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		maxFilter(imageIn,imageDims,kernDims,circleKernel,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		maxFilter(imageIn,imageDims,kernDims,circleKernel,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		maxFilter(imageIn,imageDims,kernDims,circleKernel,&imageOut,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		maxFilter(imageIn,imageDims,kernDims,circleKernel,&imageOut,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		maxFilter(imageIn,imageDims,kernDims,circleKernel,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		maxFilter(imageIn,imageDims,kernDims,circleKernel,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		maxFilter(imageIn,imageDims,kernDims,circleKernel,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}

	delete[] circleKernel;
}

std::string MexMaxFilterEllipsoid::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	if (nrhs<2 || nrhs>3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims>3)
		return "Image can have a maximum of three dimensions!";

	size_t numEl = mxGetNumberOfElements(prhs[1]);
	if (numEl!=3 || !mxIsDouble(prhs[1]))
		return "Radii must be an array of three doubles!";

	return "";
}

void MexMaxFilterEllipsoid::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
{
	inArgs.push_back("imageIn");
	inArgs.push_back("radius");
inArgs.push_back("device");
outArgs.push_back("imageOut");
}

void MexMaxFilterEllipsoid::help(std::vector<std::string>& helpLines) const
{
//\	std::string msg = "\tThis will set each pixel/voxel to the max value of an ellipsoidal neighborhood with the radii given.\n";
//\	msg += "\n";
//\	return msg;
}

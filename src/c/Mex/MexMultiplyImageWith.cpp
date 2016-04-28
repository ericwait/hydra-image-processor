#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexMultiplyTwoImages::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	int device = 0;

	if (nrhs>3)
		device = mat_to_c((int)mxGetScalar(prhs[3]));

	double factor = mxGetScalar(prhs[2]);

	Vec<size_t> imageDims, imageDims2;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		unsigned char* imageIn2;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims!=imageDims2)
			mexErrMsgTxt("Image dimensions must agree!");

		multiplyImageWith(imageIn,imageIn2,imageDims,factor,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		unsigned short* imageIn2;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims!=imageDims2)
			mexErrMsgTxt("Image dimensions must agree!");

		multiplyImageWith(imageIn,imageIn2,imageDims,factor,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		short* imageIn2;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims!=imageDims2)
			mexErrMsgTxt("Image dimensions must agree!");

		multiplyImageWith(imageIn,imageIn2,imageDims,factor,&imageOut,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		unsigned int* imageIn2;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims!=imageDims2)
			mexErrMsgTxt("Image dimensions must agree!");

		multiplyImageWith(imageIn,imageIn2,imageDims,factor,&imageOut,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		int* imageIn2;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims!=imageDims2)
			mexErrMsgTxt("Image dimensions must agree!");

		multiplyImageWith(imageIn,imageIn2,imageDims,factor,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		float* imageIn2;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims!=imageDims2)
			mexErrMsgTxt("Image dimensions must agree!");

		multiplyImageWith(imageIn,imageIn2,imageDims,factor,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		double* imageIn2;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		if (imageDims!=imageDims2)
			mexErrMsgTxt("Image dimensions must agree!");

		multiplyImageWith(imageIn,imageIn2,imageDims,factor,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexMultiplyTwoImages::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	if (nrhs<3 || nrhs>4)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t numDims1 = mxGetNumberOfDimensions(prhs[0]);
	if (numDims1>3)
		return "Image can only be either 2D or 3D!";

	size_t numDims2 = mxGetNumberOfDimensions(prhs[1]);
	if (numDims2>3)
		return "Image can have a maximum of three dimensions!";

	if (numDims1!=numDims2)
		return "Image can have a maximum of three dimensions!";

	if (!mxIsDouble(prhs[2]))
		return "Factor needs to be a double!";

	return "";
}

void MexMultiplyTwoImages::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
{
	inArgs.push_back("imageIn1");
	inArgs.push_back("imageIn2");
	inArgs.push_back("factor");
inArgs.push_back("device");
outArgs.push_back("imageOut");
}

void MexMultiplyTwoImages::help(std::vector<std::string>& helpLines) const
{
//\	std::string msg = "\tWhere factor is a multiplier.  Pixel = factor * imageIn1 * imageIn2.";
//\	msg += "\tPixel value is floored at assignment only when integer.\n";
//\	msg += "\tImageIn1 and ImageIn2 must have the same dimensions.\n";
//\	msg += "\tImageOut will have the same dimensions as the input images.\n";
//\	msg += "\n";
//\	return msg;
}

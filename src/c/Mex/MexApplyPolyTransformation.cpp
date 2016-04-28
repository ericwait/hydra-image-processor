#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexApplyPolyTransformation::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	int device = 0;

	if (nrhs>6)
		device = mat_to_c((int)mxGetScalar(prhs[6]));

	double a, b, c;
	a = mxGetScalar(prhs[1]);
	b = mxGetScalar(prhs[2]);
	c = mxGetScalar(prhs[3]);

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned char mn = std::numeric_limits<unsigned char>::lowest();
		unsigned char mx = std::numeric_limits<unsigned char>::max();

		if (nrhs>5)
			mx = MIN((unsigned char)mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = MAX((unsigned char)mxGetScalar(prhs[4]),mn);

		applyPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned short mn = std::numeric_limits<unsigned short>::lowest();
		unsigned short mx = std::numeric_limits<unsigned short>::max();

		if (nrhs>5)
			mx = MIN((unsigned short)mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = MAX((unsigned short)mxGetScalar(prhs[4]),mn);

		applyPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		short mn = std::numeric_limits<short>::lowest();
		short mx = std::numeric_limits<short>::max();

		if (nrhs>5)
			mx = MIN((short)mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = MAX((short)mxGetScalar(prhs[4]),mn);

		applyPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned int mn = std::numeric_limits<unsigned int>::lowest();
		unsigned int mx = std::numeric_limits<unsigned int>::max();

		if (nrhs>5)
			mx = MIN((unsigned int)mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = MAX((unsigned int)mxGetScalar(prhs[4]),mn);

		applyPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		int mn = std::numeric_limits<int>::lowest();
		int mx = std::numeric_limits<int>::max();

		if (nrhs>5)
			mx = MIN((int)mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = MAX((int)mxGetScalar(prhs[4]),mn);

		applyPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		float mn = std::numeric_limits<float>::lowest();
		float mx = std::numeric_limits<float>::max();

		if (nrhs>5)
			mx = MIN((float)mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = MAX((float)mxGetScalar(prhs[4]),mn);

		applyPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		double mn = (double)std::numeric_limits<double>::lowest();
		double mx = (double)std::numeric_limits<double>::max();

		if (nrhs>5)
			mx = MIN(mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = MAX(mxGetScalar(prhs[4]),mn);

		applyPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexApplyPolyTransformation::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	if (nrhs<4 && nrhs>7)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3)
		return "Image can have a maximum of three dimensions!";

	if (!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) || !mxIsDouble(prhs[3]))
		return "a,b,c all have to be doubles!";

	return "";
}

void MexApplyPolyTransformation::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
{
	inArgs.push_back("imageIn");
	inArgs.push_back("a");
	inArgs.push_back("b");
	inArgs.push_back("c");
	inArgs.push_back("min");
	inArgs.push_back("max");
	inArgs.push_back("device");

	outArgs.push_back("imageOut");
}

void MexApplyPolyTransformation::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("This returns an image with the quadradic function applied. ImageOut = a*ImageIn^2 + b*ImageIn + c");

	helpLines.push_back("\tA -- this multiplier is applied to the square of the image.");
	helpLines.push_back("\tB -- this multiplier is applied to the image.");
	helpLines.push_back("\tC -- is the constant additive.");
	helpLines.push_back("\tMin -- this is an optional parameter to clamp the output to and is useful for signed or floating point to remove negative values.");
	helpLines.push_back("\tMax -- this is an optional parameter to clamp the output to.");
	helpLines.push_back("\tDevice -- this is an optional parameter that indicates which Cuda capable device to use.");

	helpLines.push_back("\tImageOut -- this is the result of ImageOut = a*ImageIn^2 + b*ImageIn + c and is the same dimension and type as imageIn.");
}

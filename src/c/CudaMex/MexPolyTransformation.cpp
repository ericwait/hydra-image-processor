#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.cuh"

void MexPolyTransferFunction::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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

		unsigned char mn = (unsigned char)std::numeric_limits<unsigned char>::min();
		unsigned char mx = (unsigned char)std::numeric_limits<unsigned char>::max();

		if (nrhs>5)
			mx = (unsigned char)MIN(mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = (unsigned char)MAX(mxGetScalar(prhs[4]),mn);

		cPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned int mn = (unsigned int)std::numeric_limits<unsigned int>::min();
		unsigned int mx = (unsigned int)std::numeric_limits<unsigned int>::max();

		if (nrhs>5)
			mx = (unsigned int)MIN(mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = (unsigned int)MAX(mxGetScalar(prhs[4]),mn);

		cPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		int mn = (int)std::numeric_limits<int>::min();
		int mx = (int)std::numeric_limits<int>::max();

		if (nrhs>5)
			mx = (int)MIN(mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = (int)MAX(mxGetScalar(prhs[4]),mn);

		cPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		float mn = (float)std::numeric_limits<float>::min();
		float mx = (float)std::numeric_limits<float>::max();

		if (nrhs>5)
			mx = (float)MIN(mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = (float)MAX(mxGetScalar(prhs[4]),mn);

		cPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		double mn = (double)std::numeric_limits<double>::min();
		double mx = (double)std::numeric_limits<double>::max();

		if (nrhs>5)
			mx = (double)MIN(mxGetScalar(prhs[5]),mx);

		if (nrhs>4)
			mn = (double)MAX(mxGetScalar(prhs[4]),mn);

		cPolyTransferFunction(imageIn,imageDims,a,b,c,mn,mx,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexPolyTransferFunction::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<4 && nrhs>8)
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

std::string MexPolyTransferFunction::printUsage()
{
	return "imageOut = CudaMex('ApplyPolyTransformation',imageIn,a,b,c,[min],[max],[device]);";
}

std::string MexPolyTransferFunction::printHelp()
{
	std::string msg = "\ta, b, and c are the polynomial curve parameters for the transfer function which maps imageIn to imageOut.\n";
	msg += "\tmin and max are optional clamping parameters that will clamp the output values between [min,max].\n";
	msg += "\tIf min and max are not supplied, imageOut is clamped to the range of imageIn's type.\n";
	msg += "\timageOut will be the same dimension as imageIn.\n";
	msg += "\n";
	return msg;
}

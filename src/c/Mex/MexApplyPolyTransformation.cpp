#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexApplyPolyTransformation::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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

std::string MexApplyPolyTransformation::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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

std::string MexApplyPolyTransformation::printUsage()
{
	return "imageOut = CudaMex('ApplyPolyTransformation',imageIn,a,b,c,[min],[max],[device]);";
}

std::string MexApplyPolyTransformation::printHelp()
{
	std::string msg = "\ta, b, and c are the polynomial curve parameters for the transfer function which maps imageIn to imageOut.\n";
	msg += "\tmin and max are optional clamping parameters that will clamp the output values between [min,max].\n";
	msg += "\tIf min and max are not supplied, imageOut is clamped to the range of imageIn's type.\n";
	msg += "\timageOut will be the same dimension as imageIn.\n";
	msg += "\n";
	return msg;
}

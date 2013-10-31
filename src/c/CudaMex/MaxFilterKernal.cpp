#include "MexCommand.h"


void MaxFilterKernel::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<size_t> imageDims;
	ImageContainer* imageIn, * imageOut;
	HostPixelType* mexImageOut;
	setupImagePointers(prhs[0],&imageIn,&plhs[0],&mexImageOut,&imageOut);

	size_t kernDims = mxGetNumberOfDimensions(prhs[1]);
	const mwSize* DIMS = mxGetDimensions(prhs[1]);
	Vec<size_t> kernelDims(1,1,1);
	if (kernDims>1)
	{
		kernelDims.x = (size_t)DIMS[0];
		kernelDims.y = (size_t)DIMS[1];
	}
	if (kernDims>2)
		kernelDims.z = (size_t)DIMS[2];

	double* kern = (double*)mxGetData(prhs[1]);

	maxFilter(imageIn,imageOut,kernelDims,kern);
	rearange(imageOut,mexImageOut);
}

std::string MaxFilterKernel::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=2)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims>3 || imgNumDims<2)
		return "Image can only be either 2D or 3D!";

	size_t kernNumDims = mxGetNumberOfDimensions(prhs[1]);
	if (kernNumDims!=imgNumDims)
		return "structuringElement has to be same dimensional as the image!";

	return "";
}

std::string MaxFilterKernel::printUsage()
{
		return "imageOut = CudaMex('MaxFilterKernel',imageIn,kernel)";
}
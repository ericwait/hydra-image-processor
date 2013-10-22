#include "MexCommand.h"


void MaxFilterKernel::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	MexImagePixelType* imageIn, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

	size_t kernDims = mxGetNumberOfDimensions(prhs[1]);
	const mwSize* DIMS = mxGetDimensions(prhs[1]);
	Vec<unsigned int> kernelDims(1,1,1);
	if (kernDims>1)
	{
		kernelDims.x = (unsigned int)DIMS[0];
		kernelDims.y = (unsigned int)DIMS[1];
	}
	if (kernDims>2)
		kernelDims.z = (unsigned int)DIMS[2];

	double* kern = (double*)mxGetData(prhs[1]);

	maxFilter(imageIn,imageOut,imageDims,kernelDims,kern);
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
		return "imageOut = CudaMex('MaxFilterNeighborHood',imageIn,kernal)";
}
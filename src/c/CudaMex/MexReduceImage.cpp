#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"

void MexReduceImage::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	Vec<size_t> imageDims;
	HostPixelType* imageIn, * imageOut;
	CudaProcessBuffer cudaBuffer(device);
	setupImagePointers(prhs[0],&imageIn,&imageDims);

	double* reductionD = (double*)mxGetData(prhs[1]);
	Vec<double> reductionFactors(reductionD[0],reductionD[1],reductionD[2]);

	Vec<size_t> reducedDims;
	imageOut = cudaBuffer.reduceImage(imageIn, imageDims, reductionFactors, reducedDims);

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	mwSize* dims = new mwSize[numDims];

	dims[0] = reducedDims.x;
	dims[1] = reducedDims.y;
	if (numDims==3)
		dims[2] = reducedDims.z;

	plhs[0] = mxCreateNumericArray(numDims,dims,mxUINT8_CLASS,mxREAL);
	HostPixelType* mexImageOut = (HostPixelType*)mxGetData(plhs[0]);
	memcpy(mexImageOut,imageOut,sizeof(HostPixelType)*reducedDims.product());

	delete[] dims;
	delete[] imageOut;
}

std::string MexReduceImage::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<2 || nrhs>3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]))
		return "Images has to be formated as a uint8!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3 || numDims<2)
		return "Image can only be either 2D or 3D!";

	size_t numEl = mxGetNumberOfElements(prhs[1]);
	if (numEl !=3 || !mxIsDouble(prhs[1]))
		return "Reduction has to be an array of three doubles!";

	return "";
}

std::string MexReduceImage::printUsage()
{
	return "imageOut = CudaMex('ReduceImage',imageIn,[reductionFactorX,reductionFactorY,reductionFactorZ],[device]);";
}

std::string MexReduceImage::printHelp()
{
	std::string msg = "\treductionFactorX, reductionFactorY, and reductionFactorZ is the amount of\n";
	msg += "\tpixels and direction to \"collapse\" into one pixel.";
	msg += "\n";
	return msg;
}
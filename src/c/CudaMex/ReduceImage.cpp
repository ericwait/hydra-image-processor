#include "MexCommand.h"
#include "Process.h"

void ReduceImage::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	MexImagePixelType* imageIn, * processedImage, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&imageDims);

	double* reductionD = (double*)mxGetData(prhs[1]);
	Vec<double> reductionFactors(reductionD[0],reductionD[1],reductionD[2]);

	processedImage = reduceImage(imageIn,imageDims,reductionFactors);

	mxArray* argOut;
	mwSize* dims = new mwSize[3];
	dims[0] = imageDims.x;
	dims[1] = imageDims.y;
	dims[2] = imageDims.z;
	argOut = mxCreateNumericArray(3,dims,mxUINT8_CLASS,mxREAL);
	imageOut = (MexImagePixelType*)mxGetData(argOut);
	memcpy(imageOut,processedImage,sizeof(MexImagePixelType)*imageDims.product());
}

std::string ReduceImage::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=2)
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

std::string ReduceImage::printUsage()
{
	return "imageOut = CudaMex('ReduceImage',imageIn,[reductionFactor.x,reductionFactor.y,reductionFactor.z])";
}
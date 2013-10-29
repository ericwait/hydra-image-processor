#include "MexCommand.h"
#include "Process.h"

void MaximumIntensityProjection::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	HostPixelType* imageIn, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&imageDims);

	mxArray* argOut;
	mwSize* dims = new mwSize[2];
	dims[0] = imageDims.x;
	dims[1] = imageDims.y;
	argOut = mxCreateNumericArray(2,dims,mxUINT8_CLASS,mxREAL);
	imageOut = (HostPixelType*)mxGetData(argOut);

	maximumIntensityProjection(imageIn,imageOut,imageDims);
}

std::string MaximumIntensityProjection::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs!=1)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims!=3)
		return "Image has to be 3D!";

	return "";
}

std::string MaximumIntensityProjection::printUsage()
{
	return "imageOut = CudaMex('MaximumIntensityProjection',imageIn)";
}
#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"
#include "CHelpers.h"

void MexMaxFilterEllipsoid::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	Vec<size_t> imageDims;
	HostPixelType* imageIn, * imageOut;
	CudaProcessBuffer cudaBuffer(device);
	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

	double* radiiD = (double*)mxGetData(prhs[1]);

	Vec<size_t> radii((size_t)radiiD[0],(size_t)radiiD[1],(size_t)radiiD[2]);
	Vec<size_t> kernDims;
	float* circleKernel = createEllipsoidKernel(radii,kernDims);
	cudaBuffer.maxFilter(imageIn,imageDims,kernDims,circleKernel,&imageOut);

	delete[] circleKernel;
}

std::string MexMaxFilterEllipsoid::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<2 || nrhs>3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims>3 || imgNumDims<2)
		return "Image can only be either 2D or 3D!";

	size_t numEl = mxGetNumberOfElements(prhs[1]);
	if (numEl!=3 || !mxIsDouble(prhs[1]))
		return "Radii must be an array of three doubles!";

	return "";
}

std::string MexMaxFilterEllipsoid::printUsage()
{
	return "imageOut = CudaMex('MaxFilterEllipsoid',imageIn,[radiusX,radiusY,radiusZ],[device]);";
}

std::string MexMaxFilterEllipsoid::printHelp()
{
	std::string msg = "\tThis will set each pixel/voxel to the max value of an ellipsoidal neighborhood with the radii given.\n";
	msg += "\n";
	return msg;
}
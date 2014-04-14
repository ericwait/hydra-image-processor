#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"
#include "CHelpers.h"
#include "Vec.h"

void MexMinFilterKernel::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	Vec<size_t> imageDims;
	HostPixelType* imageIn, * imageOut;
	CudaProcessBuffer cudaBuffer(device);
	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

	size_t numDims = mxGetNumberOfDimensions(prhs[1]);
	const mwSize* DIMS = mxGetDimensions(prhs[1]);

	Vec<size_t> kernDims;

	if (numDims>2)
		kernDims.z = (size_t)DIMS[2];
	else
		kernDims.z = 1;

	if (numDims>1)
		kernDims.y = (size_t)DIMS[1];
	else
		kernDims.y = 1;

	if (numDims>0)
		kernDims.x = (size_t)DIMS[0];
	else
		return;

	double* matKernel;
	matKernel = (double*)mxGetData(prhs[1]);

	float* kernel = new float[kernDims.product()];
	for (int i=0; i<kernDims.product(); ++i)
		kernel[i] = (float)matKernel[i];

	cudaBuffer.minFilter(imageIn,imageDims,kernDims,kernel,&imageOut);

	delete[] kernel;
}

std::string MexMinFilterKernel::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<2 || nrhs>3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	if (!mxIsUint8(prhs[0]))
		return "Image has to be formated as a uint8!";

	size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims>3 || imgNumDims<2)
		return "Image can only be either 2-D or 3-D!";

	size_t kernDims = mxGetNumberOfDimensions(prhs[0]);
	if (kernDims<1 || kernDims>3)
		return "Kernel can only be either 1-D, 2-D, or 3-D!";

	return "";
}

std::string MexMinFilterKernel::printUsage()
{
	return "imageOut = CudaMex('MinFilterKernel',imageIn,kernel,[device]);";
}

std::string MexMinFilterKernel::printHelp()
{
	std::string msg = "\tThis will set each pixel/voxel to the min value of the neighborhood defined by the given kernel.\n";
	msg += "\n";
	return msg;
}
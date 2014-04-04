#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"

void MexMinFilterNeighborhood::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	Vec<size_t> imageDims;
	HostPixelType* imageIn, * imageOut;
	CudaProcessBuffer cudaBuffer(device);
	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

	double* nbh = (double*)mxGetData(prhs[1]);
	Vec<size_t> neighborhood((size_t)nbh[0],(size_t)nbh[1],(size_t)nbh[2]);

	cudaBuffer.minFilter(imageIn,imageDims,neighborhood,NULL,&imageOut);
}

std::string MexMinFilterNeighborhood::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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
	if (numEl!=3)
		return "Neighborhood needs to be an array of three doubles!";

	return "";
}

std::string MexMinFilterNeighborhood::printUsage()
{
	return "imageOut = CudaMex('MinFilterNeighborhood',imageIn,[NeighborhoodX,NeighborhoodY,NeighborhoodZ],[device]);";
}


std::string MexMinFilterNeighborhood::printHelp()
{
	std::string msg = "\tThis will set each pixel/voxel to the min value within the neighborhood given.\n";
	msg += "\n";
	return msg;
}
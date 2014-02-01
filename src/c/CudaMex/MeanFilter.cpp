 #include "MexCommand.h"
#include "CudaProcessBuffer.cuh"
 
 void MeanFilter::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
 	Vec<size_t> imageDims;
 	HostPixelType* imageIn, * imageOut;
	CudaProcessBuffer cudaBuffer;
 	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
 
 	double* neighborhoodD = (double*)mxGetData(prhs[1]);
 	Vec<size_t> neighborhood((int)neighborhoodD[0],(int)neighborhoodD[1],(int)neighborhoodD[2]);
 
	cudaBuffer.meanFilter(imageIn,imageDims,neighborhood,&imageOut);
 }
 
 std::string MeanFilter::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
 	if (nrhs!=2)
 		return "Incorrect number of inputs!";
 
 	if (nlhs!=1)
 		return "Requires one output!";
 
 	if (!mxIsUint8(prhs[0]))
 		return "Image has to be formated as a uint8!";
 
 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
 	if (numDims>3 || numDims<2)
 		return "Image can only be either 2D or 3D!";
 
 	size_t numEl= mxGetNumberOfElements(prhs[1]);
 	if (numEl!=3 || !mxIsDouble(prhs[1]))
 		return "Neighborhood has to be an array of three doubles!";
 
 	return "";
 }
 
 std::string MeanFilter::printUsage()
 {
 	return "imageOut = CudaMex('MeanFilter',imageIn,[NeighborhoodX,NeighborhoodY,NeighborhoodZ])";
 }
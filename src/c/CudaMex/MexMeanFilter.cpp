#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"
 
 void MexMeanFilter::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>2)
		 device = mat_to_c((int)mxGetScalar(prhs[2]));

 	Vec<size_t> imageDims;
 	HostPixelType* imageIn, * imageOut;
	CudaProcessBuffer cudaBuffer(device);
 	setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
 
 	double* neighborhoodD = (double*)mxGetData(prhs[1]);
 	Vec<size_t> neighborhood((int)neighborhoodD[1],(int)neighborhoodD[0],(int)neighborhoodD[2]);
 
	cudaBuffer.meanFilter(imageIn,imageDims,neighborhood,&imageOut);
 }
 
 std::string MexMeanFilter::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
 	if (nrhs<2 || nrhs>3)
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
 
 std::string MexMeanFilter::printUsage()
 {
 	return "imageOut = CudaMex('MeanFilter',imageIn,[NeighborhoodX,NeighborhoodY,NeighborhoodZ],[device]);";
 }

 std::string MexMeanFilter::printHelp()
 {
	 std::string msg = "\tNeighborhoodX, NeighborhoodY, and NeighborhoodZ are the directions and area to look for a given pixel.";
	 msg += "\n";
	 return msg;
 }
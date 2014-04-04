#include "MexCommand.h"
#include "CudaProcessBuffer.cuh"
 
 void MexHistogram::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>1)
		 device = mat_to_c((int)mxGetScalar(prhs[1]));

	 Vec<size_t> imageDims;
	 HostPixelType* imageIn;
	 CudaProcessBuffer cudaBuffer(device);
	 setupImagePointers(prhs[0],&imageIn,&imageDims);
 
 	int arraySize;
 	size_t* hist = cudaBuffer.createHistogram(imageIn,imageDims,arraySize);
 
 	const mwSize DIM = arraySize;
 	plhs[0] = mxCreateNumericArray(1,&DIM,mxDOUBLE_CLASS,mxREAL);
 	double* histPr = mxGetPr(plhs[0]);
 
 	for (int i=0; i<arraySize; ++i)
 		histPr[i] = double(hist[i]);
 
 	delete[] hist;
 }
 
 std::string MexHistogram::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
 	if (nrhs<1 || nrhs>2)
 		return "Incorrect number of inputs!";
 
 	if (nlhs!=1)
 		return "Requires one outputs!";
 
 	if (!mxIsUint8(prhs[0]))
 		return "Image has to be formated as a uint8!";
 
 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
 	if (numDims>3 || numDims<2)
 		return "Image can only be either 2D or 3D!";
 
 	return "";
 }
 
 std::string MexHistogram::printUsage()
 {
 	return "histogram = CudaMex('Histogram',imageIn,[device]);";
 }

 std::string MexHistogram::printHelp()
 {
	 std::string msg = "\tCreates a histogram array with 255 bins.\n";
	 msg += "\n";
	 return msg;
 }
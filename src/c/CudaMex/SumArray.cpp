// #include "MexCommand.h"
// #include "Process.h"
// 
// void SumArray::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
// {
// 	Vec<size_t> imageDims;
// 	ImageContainer* imageIn;
// 	setupImagePointers(prhs[0],&imageIn);
// 
// 	double sm = sumArray(imageIn);
// 
// 	plhs[0] = mxCreateDoubleScalar(sm);
// 
// 	delete imageIn;
// }
// 
// std::string SumArray::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
// {
// 	if (nrhs!=1)
// 		return "Incorrect number of inputs!";
// 
// 	if (nlhs!=1)
// 		return "Requires one outputs!";
// 
// 	if (!mxIsUint8(prhs[0]))
// 		return "Image has to be formated as a uint8!";
// 
// 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
// 	if (numDims>3 || numDims<2)
// 		return "Image can only be either 2D or 3D!";
// 
// 	return "";
// }
// 
// std::string SumArray::printUsage()
// {
// 	return "sum = CudaMex('SumArray',imageIn)";
// }
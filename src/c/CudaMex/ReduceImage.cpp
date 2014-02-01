// #include "MexCommand.h"
// #include "Process.h"
// 
// void ReduceImage::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
// {
// 	Vec<size_t> imageDims;
// 	HostPixelType* mexImageOut;
// 	ImageContainer* imageIn, * processedImage;
// 	setupImagePointers(prhs[0],&imageIn);
// 
// 	double* reductionD = (double*)mxGetData(prhs[1]);
// 	Vec<double> reductionFactors(reductionD[0],reductionD[1],reductionD[2]);
// 
// 	reduceImage(imageIn,&processedImage,reductionFactors);
// 
// 	mwSize* dims = new mwSize[3];
// 	dims[0] = processedImage->getHeight();
// 	dims[1] = processedImage->getWidth();
// 	dims[2] = processedImage->getDepth();
// 	plhs[0] = mxCreateNumericArray(3,dims,mxUINT8_CLASS,mxREAL);
// 	mexImageOut = (HostPixelType*)mxGetData(plhs[0]);
// 	memcpy(mexImageOut,processedImage->getConstMemoryPointer(),sizeof(HostPixelType)*processedImage->getDims().product());
// 
// 	delete processedImage;
// 	delete imageIn;
// 	delete[] dims;
// }
// 
// std::string ReduceImage::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
// {
// 	if (nrhs!=2)
// 		return "Incorrect number of inputs!";
// 
// 	if (nlhs!=1)
// 		return "Requires one output!";
// 
// 	if (!mxIsUint8(prhs[0]))
// 		return "Images has to be formated as a uint8!";
// 
// 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
// 	if (numDims>3 || numDims<2)
// 		return "Image can only be either 2D or 3D!";
// 
// 	size_t numEl = mxGetNumberOfElements(prhs[1]);
// 	if (numEl !=3 || !mxIsDouble(prhs[1]))
// 		return "Reduction has to be an array of three doubles!";
// 
// 	return "";
// }
// 
// std::string ReduceImage::printUsage()
// {
// 	return "imageOut = CudaMex('ReduceImage',imageIn,[reductionFactor.x,reductionFactor.y,reductionFactor.z])";
// }
#include "MexCommand.h"
#include "CWrappers.cuh"
#include "Vec.h"
 
 void MexAddImageWith::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>3)
		 device = mat_to_c((int)mxGetScalar(prhs[3]));

	 double additive = mxGetScalar(prhs[2]);

	 Vec<size_t> imageDims, imageDims2;
	 if (mxIsUint8(prhs[0]))
	 {
		 unsigned char* imageIn,* imageOut;
		 unsigned char* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 cAddImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else if (mxIsUint16(prhs[0]))
	 {
		 unsigned int* imageIn,* imageOut;
		 unsigned int* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 cAddImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else if (mxIsInt16(prhs[0]))
	 {
		 int* imageIn,* imageOut;
		 int* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 cAddImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else if (mxIsSingle(prhs[0]))
	 {
		 float* imageIn,* imageOut;
		 float* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 cAddImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else if (mxIsDouble(prhs[0]))
	 {
		 double* imageIn,* imageOut;
		 double* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 cAddImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else
	 {
		 mexErrMsgTxt("Image type not supported!");
	 }
 }
 
 std::string MexAddImageWith::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
 	if (nrhs<3 || nrhs>4)
 		return "Incorrect number of inputs!";
 
 	if (nlhs!=1)
 		return "Requires one output!";
 
 	size_t numDims1 = mxGetNumberOfDimensions(prhs[0]);
	if (numDims1>3)
		return "Image can have a maximum of three dimensions!";
 
 	size_t numDims2 = mxGetNumberOfDimensions(prhs[1]);
	if (numDims2>3)
		return "Image can have a maximum of three dimensions!";

	if (numDims1!=numDims2)
		return "Image dimensions must agree!";
 
 	if (!mxIsDouble(prhs[2]))
 		return "Factor needs to be a double!";
 
 	return "";
 }
 
 std::string MexAddImageWith::printUsage()
 {
	 return "imageOut = CudaMex('AddImageWith',imageIn1,imageIn2,factor,[device]);";
 }

 std::string MexAddImageWith::printHelp()
 {
	 std::string msg = "\tWhere factor is a multiplier on imageIn2.  Pixel = imageIn1 + factor*imageIn2.";
	 msg += "\tfactor is used in place of a SubImageWith (e.g. factor = -1).\n";
	 msg += "\tPixel value is floored at assignment only when integer.\n";
	 msg += "\tImageIn1 and ImageIn2 must have the same dimensions.\n";
	 msg += "\tImageOut will have the same dimensions as the input images.\n";
	 msg += "\n";
	 return msg;
 }
#include "MexCommand.h"
#include "CWrappers.h"
#include "Vec.h"
 
 void MexAddImageWith::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
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

		 addImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else if (mxIsUint16(prhs[0]))
	 {
		 unsigned short* imageIn,* imageOut;
		 unsigned short* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 addImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else if (mxIsInt16(prhs[0]))
	 {
		 short* imageIn,* imageOut;
		 short* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 addImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else if (mxIsUint32(prhs[0]))
	 {
		 unsigned int* imageIn,* imageOut;
		 unsigned int* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 addImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else if (mxIsInt32(prhs[0]))
	 {
		 int* imageIn,* imageOut;
		 int* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 addImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else if (mxIsSingle(prhs[0]))
	 {
		 float* imageIn,* imageOut;
		 float* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 addImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else if (mxIsDouble(prhs[0]))
	 {
		 double* imageIn,* imageOut;
		 double* imageIn2;
		 setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);
		 setupImagePointers(prhs[1],&imageIn2,&imageDims2);

		 if (imageDims!=imageDims2)
			 mexErrMsgTxt("Image dimensions must agree!");

		 addImageWith(imageIn,imageIn2,imageDims,additive,&imageOut,device);
	 }
	 else
	 {
		 mexErrMsgTxt("Image type not supported!");
	 }
 }
 
 std::string MexAddImageWith::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
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
 
 void MexAddImageWith::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
 {
	 inArgs.push_back("imageIn1");
	 inArgs.push_back("imageIn2");
	 inArgs.push_back("factor");
	 inArgs.push_back("device");

	 outArgs.push_back("imageOut");
 }

 void MexAddImageWith::help(std::vector<std::string>& helpLines) const
 {
	 helpLines.push_back("This takes two images and adds them together.");

	 helpLines.push_back("\tImageIn1 -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).");
	 helpLines.push_back("\tImageIn2 -- can be an image up to three dimensions and of the same type as imageIn1.");
	 helpLines.push_back("\tFactor -- this is a multiplier to the second image in the form imageOut = imageIn1 + factor*imageIn2.");
	 helpLines.push_back("\tDevice -- this is an optional parameter that indicates which Cuda capable device to use.");

	 helpLines.push_back("\timageOut -- this is the result of imageIn1 + factor*imageIn2 and will be of the same type as imageIn1.");
 }
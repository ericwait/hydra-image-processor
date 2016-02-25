#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"
 
 void MexMeanFilter::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>2)
		 device = mat_to_c((int)mxGetScalar(prhs[2]));
 
 	double* neighborhoodD = (double*)mxGetData(prhs[1]);
 	Vec<size_t> neighborhood((size_t)neighborhoodD[1],(size_t)neighborhoodD[0],(size_t)neighborhoodD[2]);
 
	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		meanFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		meanFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		meanFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		meanFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		meanFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		meanFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		meanFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
 }
 
 std::string MexMeanFilter::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
 	if (nrhs<2 || nrhs>3)
 		return "Incorrect number of inputs!";
 
 	if (nlhs!=1)
 		return "Requires one output!";
 
 	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3)
		return "Image can have a maximum of three dimensions!";
 
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
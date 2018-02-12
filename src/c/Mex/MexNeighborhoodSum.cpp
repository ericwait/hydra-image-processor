#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

 void MexNeighborhoodSum::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
 {
	 int device = 0;

	 if (nrhs>2)
		 device = mat_to_c((int)mxGetScalar(prhs[2]));

	double* nbh = (double*)mxGetData(prhs[1]);
	Vec<size_t> neighborhood((size_t)nbh[0],(size_t)nbh[1],(size_t)nbh[2]);

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn;
		unsigned short* imageOut;
		setupInputPointers(prhs[0], &imageDims, &imageIn);
		setupOutputPointers(&plhs[0], imageDims, &imageOut);

		neighborhoodSum(imageIn,imageDims,neighborhood,NULL,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn;
		unsigned int*  *imageOut;
		setupInputPointers(prhs[0], &imageDims, &imageIn);
		setupOutputPointers(&plhs[0], imageDims, &imageOut);

		neighborhoodSum(imageIn,imageDims,neighborhood,NULL,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn;
		int*  *imageOut;
		setupInputPointers(prhs[0], &imageDims, &imageIn);
		setupOutputPointers(&plhs[0], imageDims, &imageOut);

		neighborhoodSum(imageIn,imageDims,neighborhood,NULL,&imageOut,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn;
		double* imageOut;
		setupInputPointers(prhs[0], &imageDims, &imageIn);
		setupOutputPointers(&plhs[0], imageDims, &imageOut);

		neighborhoodSum(imageIn,imageDims,neighborhood,NULL,&imageOut,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn;
		double* imageOut;
		setupInputPointers(prhs[0], &imageDims, &imageIn);
		setupOutputPointers(&plhs[0], imageDims, &imageOut);

		neighborhoodSum(imageIn,imageDims,neighborhood,NULL,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn;
		double* imageOut;
		setupInputPointers(prhs[0], &imageDims, &imageIn);
		setupOutputPointers(&plhs[0], imageDims, &imageOut);

		neighborhoodSum(imageIn,imageDims,neighborhood,NULL,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn;
		double* imageOut;
		setupInputPointers(prhs[0], &imageDims, &imageIn);
		setupOutputPointers(&plhs[0], imageDims, &imageOut);

		neighborhoodSum(imageIn,imageDims,neighborhood,NULL,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
 }

 std::string MexNeighborhoodSum::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
 {
	if (nrhs<2 || nrhs>3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims>3)
		return "Image can have a maximum of three dimensions!";

	size_t numEl = mxGetNumberOfElements(prhs[1]);
	if (numEl!=3)
		return "Neighborhood needs to be an array of three doubles!";

	return "";
 }

 void MexNeighborhoodSum::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
 {
	inArgs.push_back("imageIn");
	inArgs.push_back("Neighborhood");
	inArgs.push_back("device");
	outArgs.push_back("imageOut");
 }


 void MexNeighborhoodSum::help(std::vector<std::string>& helpLines) const
 {
//\	 std::string msg = "\tThis will set each pixel/voxel to the max value within the neighborhood given.\n";
//\	 msg += "\n";
//\	 return msg;
 }
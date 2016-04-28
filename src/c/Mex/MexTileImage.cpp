#include "MexCommand.h"
#include "CWrappers.h"
#include "Vec.h"

void MexTileImage::execute(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) const
{
	int device = 0;

	if(nrhs>3)
		device = mat_to_c((int)mxGetScalar(prhs[3]));

	double* startsMat = (double*)mxGetData(prhs[1]);
	Vec<size_t> starts(size_t(mat_to_c(startsMat[1])),size_t(mat_to_c(startsMat[0])),size_t(mat_to_c(startsMat[2])));

	double* sizesMat = (double*)mxGetData(prhs[2]);
	Vec<size_t> sizes((size_t)sizesMat[1],(size_t)sizesMat[0],(size_t)sizesMat[2]);

	Vec<size_t> imageDims;
	if(mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		tileImage(imageIn,imageDims,starts,sizes,&imageOut,device);
	} else if(mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		tileImage(imageIn,imageDims,starts,sizes,&imageOut,device);
	} else if(mxIsInt16(prhs[0]))
	{
		short* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		tileImage(imageIn,imageDims,starts,sizes,&imageOut,device);
	} else if(mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		tileImage(imageIn,imageDims,starts,sizes,&imageOut,device);
	} else if(mxIsInt32(prhs[0]))
	{
		int* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		tileImage(imageIn,imageDims,starts,sizes,&imageOut,device);
	} else if(mxIsSingle(prhs[0]))
	{
		float* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		tileImage(imageIn,imageDims,starts,sizes,&imageOut,device);
	} else if(mxIsDouble(prhs[0]))
	{
		double* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		tileImage(imageIn,imageDims,starts,sizes,&imageOut,device);
	} else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexTileImage::check(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) const
{
	if(nrhs<3 || nrhs>4)
		return "Incorrect number of inputs!";

	if(nlhs!=1)
		return "Requires one output!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if(numDims>3)
		return "Image can have a maximum of three dimensions!";

	size_t numEl = mxGetNumberOfElements(prhs[1]);
	if(numEl!=3 || !mxIsDouble(prhs[1]))
		return "Starts must be an array of three doubles!";

	numEl = mxGetNumberOfElements(prhs[2]);
	if(numEl!=3 || !mxIsDouble(prhs[2]))
		return "Sizes must be an array of three doubles!";

	return "";
}

void MexTileImage::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
{
	inArgs.push_back("imageIn");
	inArgs.push_back("roiStart");
	inArgs.push_back("roiSize");
	inArgs.push_back("device");

	outArgs.push_back("imageOut");
}

void MexTileImage::help(std::vector<std::string>& helpLines) const
{
	helpLines.push_back("This will output an image that only consists of the region of interest indicated.");

	helpLines.push_back("\tImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).");
	helpLines.push_back("\tRoiStart -- this is the location of the first voxel in the region of interest (starting from the origin).  Must be the same dimension as imageIn.");
	helpLines.push_back("\tRoiSize -- this is how many voxels to include starting from roiStart. Must be the same dimension as imageIn.");
	helpLines.push_back("\tDevice -- this is an optional parameter that indicates which Cuda capable device to use.");

	helpLines.push_back("\tImageOut -- this will be an image that only contains the region of interest indicated.");
}

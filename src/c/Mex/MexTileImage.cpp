#include "MexCommand.h"
#include "CWrappers.cuh"
#include "Vec.h"

void MexTileImage::execute(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
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

std::string MexTileImage::check(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
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

std::string MexTileImage::printUsage()
{
	return "imageOut = CudaMex('TileImage',imageIn,[roiStartX,roiStartY,roiStartZ],[roiSizeX,roiSizeY,roiSizeZ],[device]);";
}

std::string MexTileImage::printHelp()
{
	std::string msg = "\tThis will output an image the same size as the input image which has the Region of Interest (ROI)\n";
	msg += "\tttitled across it.\n\n";
	return msg;
}

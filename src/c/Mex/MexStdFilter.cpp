#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexStdFilter::execute(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) const
{
	int device = 0;

	if(nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	double* neighborhoodD = (double*)mxGetData(prhs[1]);
	Vec<size_t> neighborhood((size_t)neighborhoodD[1],(size_t)neighborhoodD[0],(size_t)neighborhoodD[2]);

	Vec<size_t> imageDims;
	if(mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		stdFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	} else if(mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		stdFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	} else if(mxIsInt16(prhs[0]))
	{
		short* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		stdFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	} else if(mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		stdFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	} else if(mxIsInt32(prhs[0]))
	{
		int* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		stdFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	} else if(mxIsSingle(prhs[0]))
	{
		float* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		stdFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	} else if(mxIsDouble(prhs[0]))
	{
		double* imageIn,*imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		stdFilter(imageIn,imageDims,neighborhood,&imageOut,device);
	} else
	{
		mexErrMsgTxt("Image type not supported!");
	}
}

std::string MexStdFilter::check(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) const
{
	if(nrhs<2 || nrhs>3)
		return "Incorrect number of inputs!";

	if(nlhs!=1)
		return "Requires one output!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if(numDims>3)
		return "Image can have a maximum of three dimensions!";

	size_t numEl= mxGetNumberOfElements(prhs[1]);
	if(numEl!=3 || !mxIsDouble(prhs[1]))
		return "Neighborhood has to be an array of three doubles!";

	return "";
}

void MexStdFilter::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
{
	
	inArgs.push_back("imageIn");
	inArgs.push_back("Neighborhood");
	inArgs.push_back("device");

	outArgs.push_back("imageOut");
}

void MexStdFilter::help(std::vector<std::string>& helpLines) const
{
//\	std::string msg = "\tNeighborhoodX, NeighborhoodY, and NeighborhoodZ are the directions and area to look for a given pixel.";
//\	msg += "\n";
//\	return msg;
}

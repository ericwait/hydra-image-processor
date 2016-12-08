#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexMorphologicalClosure::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

    float* kernel = NULL;
    Vec<size_t> kernDims = FillKernel(prhs[1], &kernel);

	Vec<size_t> imageDims;
    if(mxIsLogical(prhs[0]))
    {
        bool* imageIn, *imageOut;
        setupImagePointers(prhs[0], &imageIn, &imageDims, &plhs[0], &imageOut);

        morphologicalClosure(imageIn, imageDims, kernDims, kernel, &imageOut, device);
    } else if(mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalClosure(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalClosure(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalClosure(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalClosure(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalClosure(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalClosure(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalClosure(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}

	delete[] kernel;
}

std::string MexMorphologicalClosure::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
{
	if (nrhs<2 || nrhs>3)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if (imgNumDims>3)
		return "Image can have a maximum of three dimensions!";

	size_t kernDims = mxGetNumberOfDimensions(prhs[0]);
	if (kernDims<1 || kernDims>3)
		return "Kernel can only be either 1-D, 2-D, or 3-D!";

	return "";
}

void MexMorphologicalClosure::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
{
	inArgs.push_back("imageIn");
	inArgs.push_back("kernel");
    inArgs.push_back("device");
    outArgs.push_back("imageOut");
}

void MexMorphologicalClosure::help(std::vector<std::string>& helpLines) const
{
//\	std::string msg = "\tThis will set each pixel/voxel to the max value of the neighborhood defined by the given kernel,\n";
//\	msg += "\tfollowed by a min value of the neighborhood defined by the given kernel.\n";
//\	msg += "\n";
//\	return msg;
}

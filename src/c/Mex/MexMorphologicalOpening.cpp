#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.cuh"

void MexMorphologicalOpening::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>2)
		device = mat_to_c((int)mxGetScalar(prhs[2]));

	size_t numDims = mxGetNumberOfDimensions(prhs[1]);
	const mwSize* DIMS = mxGetDimensions(prhs[1]);

	Vec<size_t> kernDims;

	if (numDims>2)
		kernDims.z = (size_t)DIMS[2];
	else
		kernDims.z = 1;

	if (numDims>1)
		kernDims.y = (size_t)DIMS[1];
	else
		kernDims.y = 1;

	if (numDims>0)
		kernDims.x = (size_t)DIMS[0];
	else
		return;

	double* matKernel;
	matKernel = (double*)mxGetData(prhs[1]);

	float* kernel = new float[kernDims.product()];
	for (int i=0; i<kernDims.product(); ++i)
		kernel[i] = (float)matKernel[i];

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalOpening(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalOpening(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalOpening(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalOpening(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalOpening(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalOpening(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		morphologicalOpening(imageIn,imageDims,kernDims,kernel,&imageOut,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}

	delete[] kernel;
}

std::string MexMorphologicalOpening::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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

std::string MexMorphologicalOpening::printUsage()
{
	return "imageOut = CudaMex('MexMorphologicalOpening',imageIn,kernel,[device]);";
}

std::string MexMorphologicalOpening::printHelp()
{
	std::string msg = "\tThis will set each pixel/voxel to the in value of the neighborhood defined by the given kernel,\n";
	msg += "\tfollowed by a max value of the neighborhood defined by the given kernel.\n";
	msg += "\n";
	return msg;
}

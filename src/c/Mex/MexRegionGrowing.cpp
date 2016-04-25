#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexRegionGrowing::execute(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
	int device = 0;
	bool allowConnection = true;

	if(nrhs>5)
		device = mat_to_c((int)mxGetScalar(prhs[5]));

	if(nrhs>4)
		allowConnection = mxGetScalar(prhs[4])>0;

	size_t numDims = mxGetNumberOfDimensions(prhs[1]);
	const mwSize* DIMS = mxGetDimensions(prhs[1]);

	Vec<size_t> kernDims;

	if(numDims>2)
		kernDims.z = (size_t)DIMS[2];
	else
		kernDims.z = 1;

	if(numDims>1)
		kernDims.y = (size_t)DIMS[1];
	else
		kernDims.y = 1;

	if(numDims>0)
		kernDims.x = (size_t)DIMS[0];
	else
		return;

	double* matKernel;
	matKernel = (double*)mxGetData(prhs[1]);

	float* kernel = new float[kernDims.product()];
	for(int i=0; i<kernDims.product(); ++i)
		kernel[i] = (float)matKernel[i];

	const mwSize* MASK_DIMS = mxGetDimensions(prhs[2]);
	bool* maskIn = (bool*)mxGetData(prhs[2]);

	plhs[0] = mxCreateLogicalArray(numDims,MASK_DIMS);
	bool* maskOut = (bool*)mxGetData(plhs[0]);
	size_t numEl = mxGetNumberOfElements(prhs[2]);
	double threshold = mxGetScalar(prhs[3]);
	
	memcpy(maskOut,maskIn,sizeof(bool)*numEl);	

	Vec<size_t> imageDims;
	if(mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		regionGrowing(imageIn,imageDims,kernDims,kernel,maskOut,threshold,allowConnection,device);
	} else if(mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		regionGrowing(imageIn,imageDims,kernDims,kernel,maskOut,threshold,allowConnection,device);
	} else if(mxIsInt16(prhs[0]))
	{
		short* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		regionGrowing(imageIn,imageDims,kernDims,kernel,maskOut,threshold,allowConnection,device);
	} else if(mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		regionGrowing(imageIn,imageDims,kernDims,kernel,maskOut,threshold,allowConnection,device);
	} else if(mxIsInt32(prhs[0]))
	{
		int* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		regionGrowing(imageIn,imageDims,kernDims,kernel,maskOut,threshold,allowConnection,device);
	} else if(mxIsSingle(prhs[0]))
	{
		float* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		regionGrowing(imageIn,imageDims,kernDims,kernel,maskOut,threshold,allowConnection,device);
	} else if(mxIsDouble(prhs[0]))
	{
		double* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		regionGrowing(imageIn,imageDims,kernDims,kernel,maskOut,threshold,allowConnection,device);
	} else
	{
		mexErrMsgTxt("Image type not supported!");
	}

	delete[] kernel;
}

std::string MexRegionGrowing::check(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
	if(nrhs<4 || nrhs>6)
		return "Incorrect number of inputs!";

	if(nlhs!=1)
		return "Requires one output!";

	size_t imgNumDims = mxGetNumberOfDimensions(prhs[0]);
	if(imgNumDims>3)
		return "Image can have a maximum of three dimensions!";

	size_t kernDims = mxGetNumberOfDimensions(prhs[1]);
	if(kernDims<1 || kernDims>3)
		return "Kernel can only be either 1-D, 2-D, or 3-D!";

	size_t maskDims = mxGetNumberOfDimensions(prhs[2]);
	if(maskDims!=imgNumDims)
		return "Mask must be the same dimension as the image!";

	if(!mxIsLogical(prhs[2]))
		return "Mask must be of logical type!";

	return "";
}

std::string MexRegionGrowing::printUsage()
{
	return "maskOut = CudaMex('RegionGrowing',imageIn,kernel,mask,threshold,[allowConnections],[device]);";
}

std::string MexRegionGrowing::printHelp()
{
	std::string msg = "\tThis will return a mask that has grown by the kernal shape for any pixels that are within a threshold of the current mask.\n";
	msg += "\n";
	return msg;
}
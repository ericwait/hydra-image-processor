#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"
#include "Defines.h"

void MexReduceImage::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	char method[255];
	ReductionMethods mthd = REDUC_MEAN;
	if (nrhs>2)
	{
		mxGetString(prhs[2],method,255);
		if (_strcmpi(method,"mean")==0)
			mthd = REDUC_MEAN;
		else if (_strcmpi(method,"median")==0)
			mthd = REDUC_MEDIAN;
		else if (_strcmpi(method,"min")==0)
			mthd = REDUC_MIN;
		else if (_strcmpi(method,"max")==0)
			mthd = REDUC_MAX;
		else
			mexErrMsgTxt("Method of reduction not supported!");
	}

	if (nrhs>3)
		device = mat_to_c((int)mxGetScalar(prhs[3]));

	double* reductionD = (double*)mxGetData(prhs[1]);
	Vec<double> reductionFactors(reductionD[0],reductionD[1],reductionD[2]);

	Vec<size_t> reducedDims;
	mwSize* dims;
	
	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		unsigned char* imageOut = reduceImage(imageIn,imageDims,reductionFactors,reducedDims,mthd,NULL,device);

		size_t numDims = mxGetNumberOfDimensions(prhs[0]);
		dims = new mwSize[numDims];

		dims[0] = reducedDims.x;
		dims[1] = reducedDims.y;
		if (numDims==3)
			dims[2] = reducedDims.z;

		plhs[0] = mxCreateNumericArray(numDims,dims,mxUINT8_CLASS,mxREAL);
		unsigned char* mexImageOut = (unsigned char*)mxGetData(plhs[0]);
		memcpy(mexImageOut,imageOut,sizeof(unsigned char)*reducedDims.product());
		delete[] imageOut;
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		unsigned short* imageOut = reduceImage(imageIn,imageDims,reductionFactors,reducedDims,mthd,NULL,device);

		size_t numDims = mxGetNumberOfDimensions(prhs[0]);
		dims = new mwSize[numDims];

		dims[0] = reducedDims.x;
		dims[1] = reducedDims.y;
		if (numDims==3)
			dims[2] = reducedDims.z;

		plhs[0] = mxCreateNumericArray(numDims,dims,mxUINT16_CLASS,mxREAL);
		unsigned short* mexImageOut = (unsigned short*)mxGetData(plhs[0]);
		memcpy(mexImageOut,imageOut,sizeof(unsigned short)*reducedDims.product());
		delete[] imageOut;
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		short* imageOut = reduceImage(imageIn,imageDims,reductionFactors,reducedDims,mthd,NULL,device);

		size_t numDims = mxGetNumberOfDimensions(prhs[0]);
		dims = new mwSize[numDims];

		dims[0] = reducedDims.x;
		dims[1] = reducedDims.y;
		if (numDims==3)
			dims[2] = reducedDims.z;

		plhs[0] = mxCreateNumericArray(numDims,dims,mxINT16_CLASS,mxREAL);
		short* mexImageOut = (short*)mxGetData(plhs[0]);
		memcpy(mexImageOut,imageOut,sizeof(short)*reducedDims.product());
		delete[] imageOut;
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		unsigned int* imageOut = reduceImage(imageIn,imageDims,reductionFactors,reducedDims,mthd,NULL,device);

		size_t numDims = mxGetNumberOfDimensions(prhs[0]);
		dims = new mwSize[numDims];

		dims[0] = reducedDims.x;
		dims[1] = reducedDims.y;
		if (numDims==3)
			dims[2] = reducedDims.z;

		plhs[0] = mxCreateNumericArray(numDims,dims,mxUINT32_CLASS,mxREAL);
		unsigned int* mexImageOut = (unsigned int*)mxGetData(plhs[0]);
		memcpy(mexImageOut,imageOut,sizeof(unsigned int)*reducedDims.product());
		delete[] imageOut;
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		int* imageOut = reduceImage(imageIn,imageDims,reductionFactors,reducedDims,mthd,NULL,device);

		size_t numDims = mxGetNumberOfDimensions(prhs[0]);
		dims = new mwSize[numDims];

		dims[0] = reducedDims.x;
		dims[1] = reducedDims.y;
		if (numDims==3)
			dims[2] = reducedDims.z;

		plhs[0] = mxCreateNumericArray(numDims,dims,mxINT32_CLASS,mxREAL);
		int* mexImageOut = (int*)mxGetData(plhs[0]);
		memcpy(mexImageOut,imageOut,sizeof(int)*reducedDims.product());
		delete[] imageOut;
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		float* imageOut = reduceImage(imageIn,imageDims,reductionFactors,reducedDims,mthd,NULL,device);

		size_t numDims = mxGetNumberOfDimensions(prhs[0]);
		dims = new mwSize[numDims];

		dims[0] = reducedDims.x;
		dims[1] = reducedDims.y;
		if (numDims==3)
			dims[2] = reducedDims.z;

		plhs[0] = mxCreateNumericArray(numDims,dims,mxSINGLE_CLASS,mxREAL);
		float* mexImageOut = (float*)mxGetData(plhs[0]);
		memcpy(mexImageOut,imageOut,sizeof(float)*reducedDims.product());
		delete[] imageOut;
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn;
		setupImagePointers(prhs[0],&imageIn,&imageDims);

		double* imageOut = reduceImage(imageIn,imageDims,reductionFactors,reducedDims,mthd,NULL,device);

		size_t numDims = mxGetNumberOfDimensions(prhs[0]);
		dims = new mwSize[numDims];

		dims[0] = reducedDims.x;
		dims[1] = reducedDims.y;
		if (numDims==3)
			dims[2] = reducedDims.z;

		plhs[0] = mxCreateNumericArray(numDims,dims,mxDOUBLE_CLASS,mxREAL);
		double* mexImageOut = (double*)mxGetData(plhs[0]);
		memcpy(mexImageOut,imageOut,sizeof(double)*reducedDims.product());
		delete[] imageOut;
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
		return;
	}

	delete[] dims;
}

std::string MexReduceImage::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<2 || nrhs>4)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one output!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3 || numDims<2)
		return "Image can only be either 2D or 3D!";

	size_t numEl = mxGetNumberOfElements(prhs[1]);
	if (numEl !=3 || !mxIsDouble(prhs[1]))
		return "Reduction has to be an array of three doubles!";

	return "";
}

std::string MexReduceImage::printUsage()
{
	return "imageOut = CudaMex('ReduceImage',imageIn,[reductionFactorX,reductionFactorY,reductionFactorZ],[method],[device]);";
}

std::string MexReduceImage::printHelp()
{
	std::string msg = "\treductionFactorX, reductionFactorY, and reductionFactorZ is the amount of\n";
	msg += "\tpixels and direction to \"collapse\" into one pixel.";
	msg += "\tThe optional parameter \"method\" can be \"median,\" \"mean,\" \"min,\" or \"max\" and will take the median or\n";
	msg += "\tof the neighborhood respectfully.";
	msg += "\n";
	return msg;
}
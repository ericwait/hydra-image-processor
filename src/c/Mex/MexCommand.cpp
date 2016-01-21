#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.cuh"

std::map<std::string,MexCommand*> MexCommand::commandList;

MexCommand::~MexCommand(){}

void MexCommand::init()
{
// 	//TODO Put every class here!
	REGISTER_COMMAND(MexAddConstant);
	REGISTER_COMMAND(MexAddImageWith);
 	REGISTER_COMMAND(MexApplyPolyTransformation);
 	REGISTER_COMMAND(MexContrastEnhancement);
	REGISTER_COMMAND(MexDeviceCount);
 	REGISTER_COMMAND(MexGaussianFilter);
 	REGISTER_COMMAND(MexHistogram);
 	REGISTER_COMMAND(MexImagePow);
	REGISTER_COMMAND(MexLinearUnmixing);
	REGISTER_COMMAND(MexMarkovRandomFieldDenoiser);
 	REGISTER_COMMAND(MexMaxFilterEllipsoid);
 	REGISTER_COMMAND(MexMaxFilterKernel);
 	REGISTER_COMMAND(MexMaxFilterNeighborhood);
 	REGISTER_COMMAND(MexMeanFilter);
 	REGISTER_COMMAND(MexMedianFilter);
 	REGISTER_COMMAND(MexMinFilterEllipsoid);
 	REGISTER_COMMAND(MexMinFilterKernel);
 	REGISTER_COMMAND(MexMinFilterNeighborhood);
	REGISTER_COMMAND(MexMinMax);
	REGISTER_COMMAND(MexMorphologicalClosure);
	REGISTER_COMMAND(MexMorphologicalOpening);
 	REGISTER_COMMAND(MexMultiplyImage);
	REGISTER_COMMAND(MexMultiplyTwoImages);
 	REGISTER_COMMAND(MexNormalizedCovariance);
 	REGISTER_COMMAND(MexNormalizedHistogram);
 	REGISTER_COMMAND(MexOtsuThresholdFilter);
 	REGISTER_COMMAND(MexOtsuThresholdValue);
 	REGISTER_COMMAND(MexReduceImage);
 	REGISTER_COMMAND(MexSumArray);
	REGISTER_COMMAND(MexSegment);
 	REGISTER_COMMAND(MexThresholdFilter);
	REGISTER_COMMAND(MexStdFilter);
	REGISTER_COMMAND(MexTileImage);
	REGISTER_COMMAND(MexVariance);
 }
 
 MexCommand* MexCommand::getCommand(std::string cmd)
 {
 	if (commandList.count(cmd)!=0)
 		return commandList[cmd];
 
 	return NULL;
 }
 
 std::string MexCommand::printUsageList()
 {
 	std::string list = "";
 	std::map<std::string,MexCommand*>::iterator it = commandList.begin();
 
 	for (; it!=commandList.end(); ++it)
 	{
 		list += it->second->printUsage();
 		list += "\n";
 	}
 	
 	return list;
 }
 
 void MexCommand::cleanUp()
 {
 	std::map<std::string,MexCommand*>::iterator it = commandList.begin();
 
 	for (; it!=commandList.end(); ++it)
 		delete it->second;
 
 	commandList.clear();
	clearDevice();
 }
 
 void MexCommand::addCommand(const std::string commandText, MexCommand* commandObject)
 {
 	commandList.insert(std::pair<std::string,MexCommand*>(commandText,commandObject));
 }

 void MexCommand::setupImagePointers(const mxArray* imageIn, unsigned char** image, Vec<size_t>* dims, mxArray** argOut/*=NULL*/, unsigned char** imageOut/*=NULL*/)
 {
	 size_t numDims = mxGetNumberOfDimensions(imageIn);
	 const mwSize* DIMS = mxGetDimensions(imageIn);

	 if (numDims>2)
		 dims->z = (size_t)DIMS[2];
	 else
		 dims->z = 1;

	 if (numDims>1)
		 dims->y = dims->y = (size_t)DIMS[1];
	 else
		 dims->y = 1;

	 dims->x = (size_t)DIMS[0];

	 *image = (unsigned char*)mxGetData(imageIn);

	 if (argOut!=NULL && imageOut!=NULL)
	 {
		 *argOut = mxCreateNumericArray(numDims,DIMS,mxUINT8_CLASS,mxREAL);
		 *imageOut = (unsigned char*)mxGetData(*argOut);
		 memset(*imageOut,0,sizeof(unsigned char)*dims->product());
	 }
 }

 void MexCommand::setupImagePointers(const mxArray* imageIn, unsigned short** image, Vec<size_t>* dims, mxArray** argOut/*=NULL*/, unsigned short** imageOut/*=NULL*/)
 {
	 size_t numDims = mxGetNumberOfDimensions(imageIn);
	 const mwSize* DIMS = mxGetDimensions(imageIn);

	 if (numDims>2)
		 dims->z = (size_t)DIMS[2];
	 else
		 dims->z = 1;

	 if (numDims>1)
		 dims->y = dims->y = (size_t)DIMS[1];
	 else
		 dims->y = 1;

	 dims->x = (size_t)DIMS[0];

	 *image = (unsigned short*)mxGetData(imageIn);

	 if (argOut!=NULL && imageOut!=NULL)
	 {
		 *argOut = mxCreateNumericArray(numDims,DIMS,mxUINT16_CLASS,mxREAL);
		 *imageOut = (unsigned short*)mxGetData(*argOut);
		 memset(*imageOut,0,sizeof(unsigned short)*dims->product());
	 }
 }

 void MexCommand::setupImagePointers(const mxArray* imageIn, short** image, Vec<size_t>* dims, mxArray** argOut/*=NULL*/, short** imageOut/*=NULL*/)
 {
	 size_t numDims = mxGetNumberOfDimensions(imageIn);
	 const mwSize* DIMS = mxGetDimensions(imageIn);

	 if (numDims>2)
		 dims->z = (size_t)DIMS[2];
	 else
		 dims->z = 1;

	 if (numDims>1)
		 dims->y = dims->y = (size_t)DIMS[1];
	 else
		 dims->y = 1;

	 dims->x = (size_t)DIMS[0];

	 *image = (short*)mxGetData(imageIn);

	 if (argOut!=NULL && imageOut!=NULL)
	 {
		 *argOut = mxCreateNumericArray(numDims,DIMS,mxINT16_CLASS,mxREAL);
		 *imageOut = (short*)mxGetData(*argOut);
		 memset(*imageOut,0,sizeof(short)*dims->product());
	 }
 }

 void MexCommand::setupImagePointers(const mxArray* imageIn, unsigned int** image, Vec<size_t>* dims, mxArray** argOut/*=NULL*/, unsigned int** imageOut/*=NULL*/)
 {
	 size_t numDims = mxGetNumberOfDimensions(imageIn);
	 const mwSize* DIMS = mxGetDimensions(imageIn);

	 if (numDims>2)
		 dims->z = (size_t)DIMS[2];
	 else
		 dims->z = 1;

	 if (numDims>1)
		 dims->y = dims->y = (size_t)DIMS[1];
	 else
		 dims->y = 1;

	 dims->x = (size_t)DIMS[0];

	 *image = (unsigned int*)mxGetData(imageIn);

	 if (argOut!=NULL && imageOut!=NULL)
	 {
		 *argOut = mxCreateNumericArray(numDims,DIMS,mxUINT32_CLASS,mxREAL);
		 *imageOut = (unsigned int*)mxGetData(*argOut);
		 memset(*imageOut,0,sizeof(unsigned int)*dims->product());
	 }
 }

 void MexCommand::setupImagePointers(const mxArray* imageIn, int** image, Vec<size_t>* dims, mxArray** argOut/*=NULL*/, int** imageOut/*=NULL*/)
 {
	 size_t numDims = mxGetNumberOfDimensions(imageIn);
	 const mwSize* DIMS = mxGetDimensions(imageIn);

	 if (numDims>2)
		 dims->z = (size_t)DIMS[2];
	 else
		 dims->z = 1;

	 if (numDims>1)
		 dims->y = dims->y = (size_t)DIMS[1];
	 else
		 dims->y = 1;

	 dims->x = (size_t)DIMS[0];

	 *image = (int*)mxGetData(imageIn);

	 if (argOut!=NULL && imageOut!=NULL)
	 {
		 *argOut = mxCreateNumericArray(numDims,DIMS,mxINT32_CLASS,mxREAL);
		 *imageOut = (int*)mxGetData(*argOut);
		 memset(*imageOut,0,sizeof(int)*dims->product());
	 }
 }

 void MexCommand::setupImagePointers(const mxArray* imageIn, float** image, Vec<size_t>* dims, mxArray** argOut/*=NULL*/, float** imageOut/*=NULL*/)
 {
	 size_t numDims = mxGetNumberOfDimensions(imageIn);
	 const mwSize* DIMS = mxGetDimensions(imageIn);

	 if (numDims>2)
		 dims->z = (size_t)DIMS[2];
	 else
		 dims->z = 1;

	 if (numDims>1)
		 dims->y = dims->y = (size_t)DIMS[1];
	 else
		 dims->y = 1;

	 dims->x = (size_t)DIMS[0];

	 *image = (float*)mxGetData(imageIn);

	 if (argOut!=NULL && imageOut!=NULL)
	 {
		 *argOut = mxCreateNumericArray(numDims,DIMS,mxSINGLE_CLASS,mxREAL);
		 *imageOut = (float*)mxGetData(*argOut);
		 memset(*imageOut,0,sizeof(float)*dims->product());
	 }
 }

 void MexCommand::setupImagePointers(const mxArray* imageIn, double** image, Vec<size_t>* dims, mxArray** argOut/*=NULL*/, double** imageOut/*=NULL*/)
 {
	 size_t numDims = mxGetNumberOfDimensions(imageIn);
	 const mwSize* DIMS = mxGetDimensions(imageIn);

	 if (numDims>2)
		 dims->z = (size_t)DIMS[2];
	 else
		 dims->z = 1;

	 if (numDims>1)
		 dims->y = dims->y = (size_t)DIMS[1];
	 else
		 dims->y = 1;

	 dims->x = (size_t)DIMS[0];

	 *image = (double*)mxGetData(imageIn);

	 if (argOut!=NULL && imageOut!=NULL)
	 {
		 *argOut = mxCreateNumericArray(numDims,DIMS,mxDOUBLE_CLASS,mxREAL);
		 *imageOut = (double*)mxGetData(*argOut);
		 memset(*imageOut,0,sizeof(double)*dims->product());
	 }
 }

#include "MexCommand.h"
#include "Vec.h"

std::map<std::string,MexCommand*> MexCommand::commandList;

MexCommand::~MexCommand(){}

void MexCommand::init()
{
// 	//TODO Put every class here!
	REGISTER_COMMAND(MexAddConstant);
	REGISTER_COMMAND(MexAddImageWith);
 	REGISTER_COMMAND(MexApplyPolyTransformation);
 	REGISTER_COMMAND(MexContrastEnhancement);
 	REGISTER_COMMAND(MexGaussianFilter);
 	REGISTER_COMMAND(MexHistogram);
 	REGISTER_COMMAND(MexImagePow);
 	REGISTER_COMMAND(MexMaxFilterEllipsoid);
 	REGISTER_COMMAND(MexMaxFilterKernel);
 	REGISTER_COMMAND(MexMaxFilterNeighborhood);
 	REGISTER_COMMAND(MexMeanFilter);
 	REGISTER_COMMAND(MexMedianFilter);
 	REGISTER_COMMAND(MexMinFilterEllipsoid);
 	REGISTER_COMMAND(MexMinFilterKernel);
 	REGISTER_COMMAND(MexMinFilterNeighborhood);
	REGISTER_COMMAND(MexMinMax);
 	REGISTER_COMMAND(MexMultiplyImage);
	REGISTER_COMMAND(MexMultiplyTwoImages);
 	REGISTER_COMMAND(MexNormalizedCovariance);
 	REGISTER_COMMAND(MexNormalizedHistogram);
 	REGISTER_COMMAND(MexOtsuThresholdFilter);
 	REGISTER_COMMAND(MexOtsuThesholdValue);
 	REGISTER_COMMAND(MexReduceImage);
 	REGISTER_COMMAND(MexSumArray);
 	REGISTER_COMMAND(MexThresholdFilter);
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
 }
 
 void MexCommand::addCommand(const std::string commandText, MexCommand* commandObject)
 {
 	commandList.insert(std::pair<std::string,MexCommand*>(commandText,commandObject));
 }
 
 void MexCommand::setupImagePointers( const mxArray* imageIn, HostPixelType** image, Vec<size_t>* dims, mxArray** argOut/*=NULL*/,
	 HostPixelType** imageOut/*=NULL*/ )
 {
 	size_t numDims = mxGetNumberOfDimensions(imageIn);
 	const mwSize* DIMS = mxGetDimensions(imageIn);
 
 	dims->x = (size_t)DIMS[0];
 	dims->y = (size_t)DIMS[1];
 	if (numDims==3)
 		dims->z = (size_t)DIMS[2];
 	else
 		dims->z = 1;
 
	*image = (HostPixelType*)mxGetData(imageIn);
 
 	if (argOut!=NULL && imageOut!=NULL)
 	{
 		*argOut = mxCreateNumericArray(numDims,DIMS,mxUINT8_CLASS,mxREAL);
 		*imageOut = (HostPixelType*)mxGetData(*argOut);
		memset(*imageOut,0,sizeof(HostPixelType)*dims->product());
 	}
 }

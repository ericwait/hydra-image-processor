#include "MexCommand.h"
#include "Vec.h"

std::map<std::string,MexCommand*> MexCommand::commandList;

MexCommand::~MexCommand(){}

void MexCommand::init()
{
// 	//TODO Put every class here!
	REGISTER_COMMAND(AddConstant);
	REGISTER_COMMAND(AddImageWith);
 	REGISTER_COMMAND(ApplyPolyTransformation);
// 	REGISTER_COMMAND(CalculateMinMax);
 	REGISTER_COMMAND(ContrastEnhancement);
 	REGISTER_COMMAND(GaussianFilter);
 	REGISTER_COMMAND(Histogram);
 	REGISTER_COMMAND(ImagePow);
// 	REGISTER_COMMAND(Mask);
 	REGISTER_COMMAND(MaxFilterEllipsoid);
 	REGISTER_COMMAND(MaxFilterKernel);
 	REGISTER_COMMAND(MaxFilterNeighborhood);
// 	REGISTER_COMMAND(MaximumIntensityProjection);
 	REGISTER_COMMAND(MeanFilter);
 	REGISTER_COMMAND(MedianFilter);
 	REGISTER_COMMAND(MinFilterEllipsoid);
 	REGISTER_COMMAND(MinFilterKernel);
 	REGISTER_COMMAND(MinFilterNeighborhood);
// 	REGISTER_COMMAND(MultiplyImage);
// 	REGISTER_COMMAND(MultiplyImageWith);
// 	REGISTER_COMMAND(NormalizedCovariance);
 	REGISTER_COMMAND(NormalizedHistogram);
 	REGISTER_COMMAND(OtsuThresholdFilter);
 	REGISTER_COMMAND(OtsuThesholdValue);
 	REGISTER_COMMAND(ReduceImage);
// 	REGISTER_COMMAND(SumArray);
 	REGISTER_COMMAND(ThresholdFilter);
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

#include "MexCommand.h"
#include "Vec.h"
#include "Process.h"

std::map<std::string,MexCommand*> MexCommand::commandList;

MexCommand::~MexCommand(){}

void MexCommand::init()
{
	//TODO Put every class here!
	REGISTER_COMMAND(AddConstant);
	REGISTER_COMMAND(AddImageWith);
	REGISTER_COMMAND(ApplyPolyTransformation);
	REGISTER_COMMAND(CalculateMinMax);
	REGISTER_COMMAND(ContrastEnhancement);
	REGISTER_COMMAND(GaussianFilter);
	REGISTER_COMMAND(Histogram);
	REGISTER_COMMAND(ImagePow);
	REGISTER_COMMAND(MaxFilterCircle);
	REGISTER_COMMAND(MaxFilterKernel);
	REGISTER_COMMAND(MaxFilterNeighborHood);
	REGISTER_COMMAND(MaximumIntensityProjection);
	REGISTER_COMMAND(MeanFilter);
	REGISTER_COMMAND(MedianFilter);
	REGISTER_COMMAND(MinFilterCircle);
	REGISTER_COMMAND(MinFilterKernel);
	REGISTER_COMMAND(MinFilterNeighborhood);
	REGISTER_COMMAND(MultiplyImage);
	REGISTER_COMMAND(MultiplyImageWith);
	REGISTER_COMMAND(NormalizedCovariance);
	REGISTER_COMMAND(NormalizedHistogram);
	REGISTER_COMMAND(OtsuThresholdFilter);
	REGISTER_COMMAND(OtsuThesholdValue);
	REGISTER_COMMAND(ReduceImage);
	REGISTER_COMMAND(SumArray);
	REGISTER_COMMAND(ThresholdFilter);
	REGISTER_COMMAND(Unmix);
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
	clear();
	std::map<std::string,MexCommand*>::iterator it = commandList.begin();

	for (; it!=commandList.end(); ++it)
		delete it->second;

	commandList.clear();
}

void MexCommand::addCommand(const std::string commandText, MexCommand* commandObject)
{
	commandList.insert(std::pair<std::string,MexCommand*>(commandText,commandObject));
}

void MexCommand::setupImagePointers( const mxArray* imageIn, MexImagePixelType** image, Vec<unsigned int>* imageDims, mxArray** argOut/*=NULL*/,
									MexImagePixelType** imageOut/*=NULL*/ )
{
	size_t numDims = mxGetNumberOfDimensions(imageIn);
	const mwSize* DIMS = mxGetDimensions(imageIn);

	*image = (MexImagePixelType*)mxGetData(imageIn);

	if (argOut!=NULL && imageOut!=NULL)
	{
		*argOut = mxCreateNumericArray(numDims,DIMS,mxUINT8_CLASS,mxREAL);
		*imageOut = (MexImagePixelType*)mxGetData(*argOut);
	}

	imageDims->x = (unsigned int)DIMS[1];
	imageDims->y = (unsigned int)DIMS[0];
	if (numDims==3)
		imageDims->z = (unsigned int)DIMS[2];
	else
		imageDims->z = 1;
}

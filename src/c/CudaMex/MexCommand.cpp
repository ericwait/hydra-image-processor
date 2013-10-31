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
	REGISTER_COMMAND(Mask);
	REGISTER_COMMAND(MaxFilterCircle);
	REGISTER_COMMAND(MaxFilterKernel);
	REGISTER_COMMAND(MaxFilterNeighborHood);
	REGISTER_COMMAND(MaximumIntensityProjection);
	REGISTER_COMMAND(MeanFilter);
	REGISTER_COMMAND(MedianFilter);
	REGISTER_COMMAND(MinFilterCircle);
	REGISTER_COMMAND(MinFilterKernel);
	REGISTER_COMMAND(MinFilterNeighborhood);
	REGISTER_COMMAND(MorphClosure);
	REGISTER_COMMAND(MorphOpening);
	REGISTER_COMMAND(MultiplyImage);
	REGISTER_COMMAND(MultiplyImageWith);
	REGISTER_COMMAND(NormalizedCovariance);
	REGISTER_COMMAND(NormalizedHistogram);
	REGISTER_COMMAND(OtsuThresholdFilter);
	REGISTER_COMMAND(OtsuThesholdValue);
	REGISTER_COMMAND(ReduceImage);
	REGISTER_COMMAND(SumArray);
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
	clearAll();
	std::map<std::string,MexCommand*>::iterator it = commandList.begin();

	for (; it!=commandList.end(); ++it)
		delete it->second;

	commandList.clear();
}

void MexCommand::addCommand(const std::string commandText, MexCommand* commandObject)
{
	commandList.insert(std::pair<std::string,MexCommand*>(commandText,commandObject));
}

void MexCommand::setupImagePointers( const mxArray* imageIn, ImageContainer** image, mxArray** argOut/*=NULL*/, HostPixelType** mexImageOut/*=NULL*/, ImageContainer** imageOut/*=NULL*/ )
{
	size_t numDims = mxGetNumberOfDimensions(imageIn);
	const mwSize* DIMS = mxGetDimensions(imageIn);

	Vec<size_t> imageDims;
	imageDims.x = (size_t)DIMS[1];
	imageDims.y = (size_t)DIMS[0];
	if (numDims==3)
		imageDims.z = (size_t)DIMS[2];
	else
		imageDims.z = 1;

	*image = new ImageContainer((HostPixelType*)mxGetData(imageIn),imageDims,true);

	if (argOut!=NULL && mexImageOut!=NULL && imageOut!=NULL)
	{
		*argOut = mxCreateNumericArray(numDims,DIMS,mxUINT8_CLASS,mxREAL);
		*mexImageOut = (HostPixelType*)mxGetData(*argOut);
		*imageOut = new ImageContainer(imageDims);
	}
}

void MexCommand::rearange( ImageContainer* image, HostPixelType* mexImage )
{
	Vec<size_t> curIdx(0,0,0);
	for (curIdx.z=0; curIdx.z<image->getDepth(); ++curIdx.z)
	{
		for (curIdx.y=0; curIdx.y<image->getHeight(); ++curIdx.y)
		{
			for (curIdx.x=0; curIdx.x<image->getWidth(); ++curIdx.x)
			{
				mexImage[image->getDims().linearAddressAt(curIdx,true)] = (*image)[curIdx];
			}
		}
	}
}

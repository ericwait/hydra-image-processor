#include "MexCommand.h"
#include "Process.h"

void MaximumIntensityProjection::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	Vec<unsigned int> imageDims;
	MexImagePixelType* imageIn, * imageOut;
	setupImagePointers(prhs[0],&imageIn,&imageDims);
// TODO
// 	if (argOut!=NULL && imageOut!=NULL)
// 	{
// 		*argOut = mxCreateNumericArray(numDims,DIMS,mxUINT8_CLASS,mxREAL);
// 		*imageOut = (MexImagePixelType*)mxGetData(*argOut);
// 	}

	maximumIntensityProjection(imageIn,imageOut,imageDims);
}

std::string MaximumIntensityProjection::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{

}

std::string MaximumIntensityProjection::printUsage()
{

}
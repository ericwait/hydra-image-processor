#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"

void MexNormalizedHistogram::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	int device = 0;

	if (nrhs>4)
		device = mat_to_c((int)mxGetScalar(prhs[4]));

	unsigned int arraySize = (unsigned int)mxGetScalar(prhs[1]);
	double* hist;

	Vec<size_t> imageDims;
	if (mxIsUint8(prhs[0]))
	{
		unsigned char* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned char mn = std::numeric_limits<unsigned char>::lowest();
		unsigned char mx = std::numeric_limits<unsigned char>::max();

		if (nrhs>2)
			mn = MAX(mn,(unsigned char)mxGetScalar(prhs[2]));

		if (nrhs>3)
			mx = MIN(mx,(unsigned char)mxGetScalar(prhs[3]));

		hist = normalizeHistogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned short mn = std::numeric_limits<unsigned short>::lowest();
		unsigned short mx = std::numeric_limits<unsigned short>::max();

		if (nrhs>2)
			mn = MAX(mn,(unsigned short)mxGetScalar(prhs[2]));

		if (nrhs>3)
			mx = MIN(mx,(unsigned short)mxGetScalar(prhs[3]));

		hist = normalizeHistogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		short* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		short mn = std::numeric_limits<short>::lowest();
		short mx = std::numeric_limits<short>::max();

		if (nrhs>2)
			mn = MAX(mn,(short)mxGetScalar(prhs[2]));

		if (nrhs>3)
			mx = MIN(mx,(short)mxGetScalar(prhs[3]));

		hist = normalizeHistogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else if (mxIsUint32(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned int mn = std::numeric_limits<unsigned int>::lowest();
		unsigned int mx = std::numeric_limits<unsigned int>::max();

		if (nrhs>2)
			mn = MAX(mn,(unsigned int)mxGetScalar(prhs[2]));

		if (nrhs>3)
			mx = MIN(mx,(unsigned int)mxGetScalar(prhs[3]));

		hist = normalizeHistogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else if (mxIsInt32(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		int mn = std::numeric_limits<int>::lowest();
		int mx = std::numeric_limits<int>::max();

		if (nrhs>2)
			mn = MAX(mn,(int)mxGetScalar(prhs[2]));

		if (nrhs>3)
			mx = MIN(mx,(int)mxGetScalar(prhs[3]));

		hist = normalizeHistogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else if (mxIsSingle(prhs[0]))
	{
		float* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		float mn = std::numeric_limits<float>::lowest();
		float mx = std::numeric_limits<float>::max();

		if (nrhs>2)
			mn = MAX(mn,(float)mxGetScalar(prhs[2]));

		if (nrhs>3)
			mx = MIN(mx,(float)mxGetScalar(prhs[3]));

		hist = normalizeHistogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else if (mxIsDouble(prhs[0]))
	{
		double* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		double mn = std::numeric_limits<double>::lowest();
		double mx = std::numeric_limits<double>::max();

		if (nrhs>2)
			mn = MAX(mn,mxGetScalar(prhs[2]));

		if (nrhs>3)
			mx = MIN(mx,mxGetScalar(prhs[3]));

		hist = normalizeHistogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
		return;
	}

	const mwSize DIM = arraySize;
	plhs[0] = mxCreateNumericArray(1,&DIM,mxDOUBLE_CLASS,mxREAL);
	double* histPr = mxGetPr(plhs[0]);

	for (unsigned int i=0; i<arraySize; ++i)
		histPr[i] = hist[i];

	delete[] hist;
}

std::string MexNormalizedHistogram::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{
	if (nrhs<2 || nrhs>5)
		return "Incorrect number of inputs!";

	if (nlhs!=1)
		return "Requires one outputs!";

	size_t numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims>3)
		return "Image can have a maximum of three dimensions!";

	return "";
}

std::string MexNormalizedHistogram::printUsage()
{
	return "histogram = CudaMex('NormalizedHistogram',imageIn,numBins,[min],[max],[device]);";
}

std::string MexNormalizedHistogram::printHelp()
{
	std::string msg = "\tCreates a histogram array with numBins bins\n";
	msg += "\tIf min/max is not provided, the min/max of the type is used.";
	msg += "\tEach bin is normalized over the total number of pixel/voxels.\n";
	msg = "\n";
	return msg;
}
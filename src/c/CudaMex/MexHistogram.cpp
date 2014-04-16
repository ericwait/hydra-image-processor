#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.cuh"
 
 void MexHistogram::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
 {
	 int device = 0;

	 if (nrhs>4)
		 device = mat_to_c((int)mxGetScalar(prhs[4]));

 	unsigned int arraySize = (unsigned int)mxGetScalar(prhs[1]);
	size_t* hist;

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
		
		hist = cHistogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else if (mxIsUint16(prhs[0]))
	{
		unsigned int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		unsigned int mn = std::numeric_limits<unsigned int>::lowest();
		unsigned int mx = std::numeric_limits<unsigned int>::max();

		if (nrhs>2)
			mn = MAX(mn,(unsigned int)mxGetScalar(prhs[2]));

		if (nrhs>3)
			mx = MIN(mx,(unsigned int)mxGetScalar(prhs[3]));

		hist = cHistogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else if (mxIsInt16(prhs[0]))
	{
		int* imageIn,* imageOut;
		setupImagePointers(prhs[0],&imageIn,&imageDims,&plhs[0],&imageOut);

		int mn = std::numeric_limits<int>::lowest();
		int mx = std::numeric_limits<int>::max();

		if (nrhs>2)
			mn = MAX(mn,(int)mxGetScalar(prhs[2]));

		if (nrhs>3)
			mx = MIN(mx,(int)mxGetScalar(prhs[3]));

		hist = cHistogram(imageIn,imageDims,arraySize,mn,mx,device);
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

		hist = cHistogram(imageIn,imageDims,arraySize,mn,mx,device);
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

		hist = cHistogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
	}
 
 	const mwSize DIM = arraySize;
 	plhs[0] = mxCreateNumericArray(1,&DIM,mxDOUBLE_CLASS,mxREAL);
 	double* histPr = mxGetPr(plhs[0]);
 
 	for (unsigned int i=0; i<arraySize; ++i)
 		histPr[i] = double(hist[i]);
 
 	delete[] hist;
 }
 
 std::string MexHistogram::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
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
 
 std::string MexHistogram::printUsage()
 {
 	return "histogram = CudaMex('Histogram',imageIn,numBins,[min],[max],[device]);";
 }

 std::string MexHistogram::printHelp()
 {
	 std::string msg = "\tCreates a histogram array with numBins bins between min/max values.\n";
	 msg += "\tIf min/max is not provided, the min/max of the type is used.";
	 msg += "\n";
	 return msg;
 }
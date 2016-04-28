#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"
 
 void MexHistogram::execute( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
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
		
		hist = histogram(imageIn,imageDims,arraySize,mn,mx,device);
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

		hist = histogram(imageIn,imageDims,arraySize,mn,mx,device);
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

		hist = histogram(imageIn,imageDims,arraySize,mn,mx,device);
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

		hist = histogram(imageIn,imageDims,arraySize,mn,mx,device);
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

		hist = histogram(imageIn,imageDims,arraySize,mn,mx,device);
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

		hist = histogram(imageIn,imageDims,arraySize,mn,mx,device);
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

		hist = histogram(imageIn,imageDims,arraySize,mn,mx,device);
	}
	else
	{
		mexErrMsgTxt("Image type not supported!");
		return;
	}
 
 	const mwSize DIM = arraySize;
 	plhs[0] = mxCreateNumericArray(1,&DIM,mxUINT64_CLASS,mxREAL);
 	size_t* histPr = (size_t*)mxGetPr(plhs[0]);
 
 	for (unsigned int i=0; i<arraySize; ++i)
 		histPr[i] = hist[i];
 
 	delete[] hist;
 }
 
 std::string MexHistogram::check( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) const
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
 
 void MexHistogram::usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const
 {
	 inArgs.push_back("imageIn");
	 inArgs.push_back("numBins");
	 inArgs.push_back("min");
	 inArgs.push_back("max");
	 inArgs.push_back("device");

	 outArgs.push_back("histogram");
 }

 void MexHistogram::help(std::vector<std::string>& helpLines) const
 {
	 helpLines.push_back("Creates a histogram array with numBins bins between min/max values.");

	 helpLines.push_back("\tImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).");
	 helpLines.push_back("\tNumBins -- number of bins that the histogram should partition the signal into.");
	 helpLines.push_back("\tMin -- this is the minimum value for the histogram.");
	 helpLines.push_back("\t\tIf min is not provided, the min of the image type is used.");
	 helpLines.push_back("\tMax -- this is the maximum value for the histogram.");
	 helpLines.push_back("\t\tIf min is not provided, the min of the image type is used.");
	 helpLines.push_back("\tDevice -- this is an optional parameter that indicates which Cuda capable device to use.");

	 helpLines.push_back("\tImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.");
 }
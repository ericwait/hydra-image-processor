#pragma once
#include <map>
#include <string>
#include "mex.h"
#include "Vec.h"

#define REGISTER_COMMAND(cmd) {MexCommand::addCommand(#cmd,(MexCommand*)new cmd());}

class MexCommand
{
public:
	virtual ~MexCommand();
	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) = 0;
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) = 0;
	virtual std::string printUsage() = 0;
	virtual std::string printHelp() = 0;

	static void init();
	static bool needsInit(){return commandList.empty();}
	static MexCommand* getCommand(std::string cmd);
	static std::string printUsageList();
	static void cleanUp();
protected:
	MexCommand(){}
	static void addCommand(const std::string commandText, MexCommand* commandObject);

	void setupImagePointers( const mxArray* imageIn, unsigned char** image, Vec<size_t>* dims, mxArray** argOut=NULL,
		unsigned char** imageOut=NULL);
	void setupImagePointers( const mxArray* imageIn, unsigned short** image, Vec<size_t>* dims, mxArray** argOut=NULL,
		unsigned short** imageOut=NULL);
	void setupImagePointers( const mxArray* imageIn, short** image, Vec<size_t>* dims, mxArray** argOut=NULL, short** imageOut=NULL);
	void setupImagePointers( const mxArray* imageIn, unsigned int** image, Vec<size_t>* dims, mxArray** argOut=NULL,
		unsigned int** imageOut=NULL);
	void setupImagePointers( const mxArray* imageIn, int** image, Vec<size_t>* dims, mxArray** argOut=NULL, int** imageOut=NULL);
	void setupImagePointers( const mxArray* imageIn, float** image, Vec<size_t>* dims, mxArray** argOut=NULL, float** imageOut=NULL);
	void setupImagePointers( const mxArray* imageIn, double** image, Vec<size_t>* dims, mxArray** argOut=NULL, double** imageOut=NULL);

private:
	static std::map<std::string,MexCommand*> commandList;
};
 
 class MexAddConstant : MexCommand
 {
 public:
 	MexAddConstant(){}
 	virtual ~MexAddConstant(){}
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
	virtual std::string printHelp();
 };
 
 class MexAddImageWith : MexCommand
 {
 public:
 	MexAddImageWith(){}
 	virtual ~MexAddImageWith(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
	virtual std::string printHelp();
 };
 
class MexApplyPolyTransformation : MexCommand
{
public:
	MexApplyPolyTransformation(){}
	virtual ~MexApplyPolyTransformation(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

class MexContrastEnhancement : MexCommand
{
public:
	MexContrastEnhancement(){}
	virtual ~MexContrastEnhancement(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

 class MexGaussianFilter : MexCommand
 {
 public:
 	MexGaussianFilter(){}
 	virtual ~MexGaussianFilter(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
	virtual std::string printHelp();
 };

 class MexMinMax: MexCommand
 {
 public:
	 MexMinMax(){}
	 virtual ~MexMinMax(){}

	 virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	 virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	 virtual std::string printUsage();
	 virtual std::string printHelp();
 };

class MexHistogram : MexCommand
{
public:
	MexHistogram(){}
	virtual ~MexHistogram(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
virtual std::string printHelp();
};

class MexImagePow : MexCommand
{
public:
	MexImagePow(){}
	virtual ~MexImagePow(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

class MexMarkovRandomFieldDenoiser : MexCommand
{
public:
	MexMarkovRandomFieldDenoiser(){}
	virtual ~MexMarkovRandomFieldDenoiser(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

class MexMaxFilterEllipsoid : MexCommand
{
public:
	MexMaxFilterEllipsoid(){}
	virtual ~MexMaxFilterEllipsoid(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

 class MexMaxFilterKernel : MexCommand
 {
 public:
 	MexMaxFilterKernel(){}
 	virtual ~MexMaxFilterKernel(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
 virtual std::string printHelp();
 };
 
 class MexMaxFilterNeighborhood : MexCommand
 {
 public:
 	MexMaxFilterNeighborhood(){}
 	virtual ~MexMaxFilterNeighborhood(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
 virtual std::string printHelp();
 };
 
 class MexMeanFilter : MexCommand
 {
 public:
 	MexMeanFilter(){}
 	virtual ~MexMeanFilter(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
	virtual std::string printHelp();
 };

class MexMedianFilter : MexCommand
{
public:
	MexMedianFilter(){}
	virtual ~MexMedianFilter(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

 class MexMinFilterEllipsoid : MexCommand
 {
 public:
 	MexMinFilterEllipsoid(){}
 	virtual ~MexMinFilterEllipsoid(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
virtual std::string printHelp();
 };
 
 class MexMinFilterKernel : MexCommand
 {
 public:
 	MexMinFilterKernel(){}
 	virtual ~MexMinFilterKernel(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
virtual std::string printHelp();
 };
 
 class MexMinFilterNeighborhood : MexCommand
 {
 public:
 	MexMinFilterNeighborhood(){}
 	virtual ~MexMinFilterNeighborhood(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
virtual std::string printHelp();
 };

 class MexMorphologicalClosure : MexCommand
 {
 public:
	 MexMorphologicalClosure(){}
	 virtual ~MexMorphologicalClosure(){}

	 virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	 virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	 virtual std::string printUsage();
	 virtual std::string printHelp();
 };
 
 class MexMorphologicalOpening : MexCommand
 {
 public:
	 MexMorphologicalOpening(){}
	 virtual ~MexMorphologicalOpening(){}

	 virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	 virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	 virtual std::string printUsage();
	 virtual std::string printHelp();
 };

 class MexMultiplyImage : MexCommand
 {
 public:
 	MexMultiplyImage(){}
 	virtual ~MexMultiplyImage(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
virtual std::string printHelp();
 };
 
 class MexMultiplyTwoImages : MexCommand
 {
 public:
 	MexMultiplyTwoImages(){}
 	virtual ~MexMultiplyTwoImages(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
virtual std::string printHelp();
 };
 
class MexNormalizedCovariance : MexCommand
{
public:
	MexNormalizedCovariance(){}
	virtual ~MexNormalizedCovariance(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
virtual std::string printHelp();
};

class MexNormalizedHistogram : MexCommand
{
public:
	MexNormalizedHistogram(){}
	virtual ~MexNormalizedHistogram(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};
 
class MexOtsuThresholdFilter : MexCommand
{
public:
	MexOtsuThresholdFilter(){}
	virtual ~MexOtsuThresholdFilter(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

class MexOtsuThresholdValue : MexCommand
{
public:
	MexOtsuThresholdValue(){}
	virtual ~MexOtsuThresholdValue(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

class MexSumArray : MexCommand
{
public:
	MexSumArray(){}
	virtual ~MexSumArray(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

class MexSegment : MexCommand
{
public:
	MexSegment(){}
	virtual ~MexSegment(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

 class MexReduceImage : MexCommand
 {
 public:
 	MexReduceImage(){}
 	virtual ~MexReduceImage(){}
 
 	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
 	virtual std::string printUsage();
	virtual std::string printHelp();
 };

class MexThresholdFilter : MexCommand
{
public:
	MexThresholdFilter(){}
	virtual ~MexThresholdFilter(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

class MexVariance : MexCommand
{
public:
	MexVariance(){}
	virtual ~MexVariance(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
	virtual std::string printHelp();
};

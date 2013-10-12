#pragma once
#include <map>
#include <string>
#include "mex.h"
#include "Vec.h"
#include "Process.h"

#define REGISTER_COMMAND(cmd) {MexCommand::addCommand(#cmd,(MexCommand*)new cmd());}

class MexCommand
{
public:
	virtual ~MexCommand();
	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) = 0;
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) = 0;
	virtual std::string printUsage() = 0;

	static void init();
	static bool needsInit(){return commandList.empty();}
	static MexCommand* getCommand(std::string cmd);
	static std::string printUsageList();
	static void cleanUp();
protected:
	MexCommand(){}
	static void addCommand(const std::string commandText, MexCommand* commandObject);
	void setupImagePointers( const mxArray* imageIn, MexImagePixelType** image, Vec<unsigned int>* imageDims, mxArray** argOut=NULL,
		MexImagePixelType** imageOut=NULL);

private:
	static std::map<std::string,MexCommand*> commandList;
};

class AddConstant : MexCommand
{
public:
	AddConstant(){}
	virtual ~AddConstant(){}
	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class AddImageWith : MexCommand
{
public:
	AddImageWith(){}
	virtual ~AddImageWith(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class ApplyPolyTransformation : MexCommand
{
public:
	ApplyPolyTransformation(){}
	virtual ~ApplyPolyTransformation(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class CalculateMinMax : MexCommand
{
public:
	CalculateMinMax(){}
	virtual ~CalculateMinMax(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class ContrastEnhancement : MexCommand
{
public:
	ContrastEnhancement(){}
	virtual ~ContrastEnhancement(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class GaussianFilter : MexCommand
{
public:
	GaussianFilter(){}
	virtual ~GaussianFilter(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class MaxFilter : MexCommand
{
public:
	MaxFilter(){}
	virtual ~MaxFilter(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class MaximumIntensityProjection : MexCommand
{
public:
	MaximumIntensityProjection(){}
	virtual ~MaximumIntensityProjection(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class MeanFilter : MexCommand
{
public:
	MeanFilter(){}
	virtual ~MeanFilter(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class MedianFilter : MexCommand
{
public:
	MedianFilter(){}
	virtual ~MedianFilter(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class MinFilter
{
public:
	MinFilter(){}
	virtual ~MinFilter(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class MultiplyImage
{
public:
	MultiplyImage(){}
	virtual ~MultiplyImage(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class MultiplyImageWith
{
public:
	MultiplyImageWith(){}
	virtual ~MultiplyImageWith(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class ImagePow : MexCommand
{
public:
	ImagePow(){}
	virtual ~ImagePow(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class SumArray
{
public:
	SumArray(){}
	virtual ~SumArray(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class ReduceImage
{
public:
	ReduceImage(){}
	virtual ~ReduceImage(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class ThresholdFilter
{
public:
	ThresholdFilter(){}
	virtual ~ThresholdFilter(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

class Unmix
{
public:
	Unmix(){}
	virtual ~Unmix(){}

	virtual void execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
	virtual std::string printUsage();
};

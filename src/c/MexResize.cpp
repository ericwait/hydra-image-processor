#include "MexCommand.h"
#include "Vec.h"
#include "CWrappers.h"
#include "Defines.h"

void MexResize::execute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
    Vec<size_t> imageInDims(0, 0, 0);
    setupDims(prhs[0], &imageInDims);
    Vec<double> reductionFactors(0.0, 0.0, 0.0);
    if(!mxIsEmpty(prhs[1]))
    {
        double* reductionD = (double*)mxGetData(prhs[1]);
        reductionFactors = Vec<double>(reductionD[0], reductionD[1], reductionD[2]);
    }

    Vec<size_t> imageOutDims(0, 0, 0);
    if(!mxIsEmpty(prhs[2]))
    {
        double* outDim = (double*)mxGetData(prhs[2]);
        imageOutDims = Vec<size_t>(outDim[0], outDim[1], outDim[2]);

        reductionFactors = Vec<double>(imageOutDims)/Vec<double>(imageInDims);
    }

    char method[36];
    ReductionMethods mthd = REDUC_MEAN;
    mxGetString(prhs[3], method, 255);
    if(_strcmpi(method, "mean")==0)
        mthd = REDUC_MEAN;
    else if(_strcmpi(method, "median")==0)
        mthd = REDUC_MEDIAN;
    else if(_strcmpi(method, "min")==0)
        mthd = REDUC_MIN;
    else if(_strcmpi(method, "max")==0)
        mthd = REDUC_MAX;
    else if(_strcmpi(method, "gaussian")==0)
        mthd = REDUC_GAUS;
    else
        mexErrMsgTxt("Method of resize not supported!");

    int device = 0;
    if(nrhs>4)
        device = mat_to_c((int)mxGetScalar(prhs[4]));

    if(mxIsLogical(prhs[0]))
    {
        bool* imageIn, *imageOut;
        setupInputPointers(prhs[0], &imageInDims, &imageIn);
        imageOutDims = Vec<size_t>(Vec<double>(imageInDims)*reductionFactors);
        setupOutputPointers(&(plhs[0]), imageOutDims, &imageOut);

        resize(imageIn, imageInDims, reductionFactors, imageOutDims, mthd, &imageOut, device);
    } else if(mxIsUint8(prhs[0]))
    {
        unsigned char* imageIn, *imageOut;
        setupInputPointers(prhs[0], &imageInDims, &imageIn);
        imageOutDims = imageInDims*reductionFactors;
        setupOutputPointers(&(plhs[0]), imageOutDims, &imageOut);

        resize(imageIn, imageInDims, reductionFactors, imageOutDims, mthd, &imageOut, device);
    } else if(mxIsUint16(prhs[0]))
    {
        unsigned short* imageIn, *imageOut;
        setupInputPointers(prhs[0], &imageInDims, &imageIn);
        imageOutDims = Vec<size_t>(Vec<double>(imageInDims)*reductionFactors);
        setupOutputPointers(&(plhs[0]), imageOutDims, &imageOut);
        
        resize(imageIn, imageInDims, reductionFactors, imageOutDims, mthd, &imageOut, device);
    } else if(mxIsInt16(prhs[0]))
    {
        short* imageIn, *imageOut;
        setupInputPointers(prhs[0], &imageInDims, &imageIn);
        imageOutDims = Vec<size_t>(Vec<double>(imageInDims)*reductionFactors);
        setupOutputPointers(&(plhs[0]), imageOutDims, &imageOut);
        
        resize(imageIn, imageInDims, reductionFactors, imageOutDims, mthd, &imageOut, device);
    } else if(mxIsUint32(prhs[0]))
    {
        unsigned int* imageIn, *imageOut;
        setupInputPointers(prhs[0], &imageInDims, &imageIn);
        imageOutDims = Vec<size_t>(Vec<double>(imageInDims)*reductionFactors);
        setupOutputPointers(&(plhs[0]), imageOutDims, &imageOut);
        
        resize(imageIn, imageInDims, reductionFactors, imageOutDims, mthd, &imageOut, device);
    } else if(mxIsInt32(prhs[0]))
    {
        int* imageIn, *imageOut;
        setupInputPointers(prhs[0], &imageInDims, &imageIn);
        imageOutDims = Vec<size_t>(Vec<double>(imageInDims)*reductionFactors);
        setupOutputPointers(&(plhs[0]), imageOutDims, &imageOut);
        
        resize(imageIn, imageInDims, reductionFactors, imageOutDims, mthd, &imageOut, device);
    } else if(mxIsSingle(prhs[0]))
    {
        float* imageIn, *imageOut;
        setupInputPointers(prhs[0], &imageInDims, &imageIn);
        imageOutDims = Vec<size_t>(Vec<double>(imageInDims)*reductionFactors);
        setupOutputPointers(&(plhs[0]), imageOutDims, &imageOut);
        
        resize(imageIn, imageInDims, reductionFactors, imageOutDims, mthd, &imageOut, device);
    } else if(mxIsDouble(prhs[0]))
    {
        double* imageIn, *imageOut;
        setupInputPointers(prhs[0], &imageInDims, &imageIn);
        imageOutDims = Vec<size_t>(Vec<double>(imageInDims)*reductionFactors);
        setupOutputPointers(&(plhs[0]), imageOutDims, &imageOut);
        
        resize(imageIn, imageInDims, reductionFactors, imageOutDims, mthd, &imageOut, device);
    } else
    {
        mexErrMsgTxt("Image type not supported!");
    }
}

std::string MexResize::check(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) const
{
    if (nrhs<3 || nrhs>5)
        return "Incorrect number of inputs!";

    size_t numDims = mxGetNumberOfDimensions(prhs[0]);
    if(numDims>3||numDims<2)
        return "Image can only be either 2D or 3D!";

    size_t numElFac = mxGetNumberOfElements(prhs[1]);
    size_t numElSize = mxGetNumberOfElements(prhs[2]);

    if(numElFac==0&&numElSize==0)
        return "Either ResizeFactor of ExplicitSize must be set.";

    if(!mxIsEmpty(prhs[1]) && (numElFac!=3||!mxIsDouble(prhs[1])))
        return "Resize amounts have to be an array of three doubles!";

    if(!mxIsEmpty(prhs[2]) &&(numElSize!=3||!mxIsDouble(prhs[2])))
        return "ExplicitSize amounts have to be an array of three doubles!";

    return "";
}

void MexResize::usage(std::vector<std::string>& outArgs, std::vector<std::string>& inArgs) const
{
    inArgs.push_back("imageIn");
    inArgs.push_back("resizeFactor");
    inArgs.push_back("explicitSize");
    inArgs.push_back("method");
    inArgs.push_back("device");

    outArgs.push_back("imageOut");
}

void MexResize::help(std::vector<std::string>& helpLines) const
{
    helpLines.push_back("Resizes image using various methods.");

    helpLines.push_back("\tImageIn -- can be an image up to three dimensions and of type (logical,uint8,int8,uint16,int16,uint32,int32,single,double).");
    helpLines.push_back("\tResizeFactor_rcz -- This represents the output size relative to input (r,c,z). Values less than one but greater than zero will reduce the image.");
    helpLines.push_back("\t\tValues greater than one will enlarge the image. If this is an empty array, it will be calculated from the explicit resize.");
    helpLines.push_back("\t\t\tIf both resizeFactor and explicitSize are both set, the explicitSize will be used.");
    helpLines.push_back("\tExplicitSize_rcz -- This is the size that the output should be (r,c,z). If this is an empty array, then the resize factor is used.");
    helpLines.push_back("\t\t\tIf both resizeFactor and explicitSize are both set, the explicitSize will be used.");
    helpLines.push_back("\tMethod -- This is the neighborhood operation to apply when resizing (mean, median, min, max, gaussian).");
    helpLines.push_back("\tDevice -- this is an optional parameter that indicates which Cuda capable device to use.");

    helpLines.push_back("\tImageOut -- This will be a resize image the same type as the input image.");
}

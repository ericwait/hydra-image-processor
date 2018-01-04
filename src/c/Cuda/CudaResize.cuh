#ifndef CUDA_RESIZE_CUH
#define CUDA_RESIZE_CUH

#include "CudaImageContainer.cuh"
#include "CudaMedianFilter.cuh"

#include "Vec.h"
#include <vector>
#include "ImageChunk.cuh"
#include "CudaImageContainerClean.cuh"
#include "Defines.h"
#include "CudaUtilities.cuh"

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif


template <class PixelType>
__global__ void cudaMeanResize(CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, Vec<float> hostResizeFactors, Vec<int> hostKernelDims, PixelType minVal, PixelType maxVal)
{
    Vec<float> resizeFactors = hostResizeFactors;
    Vec<int> kernelDims = hostKernelDims;
    Vec<int> coordinateOut;
    coordinateOut.x = threadIdx.x+blockIdx.x * blockDim.x;
    coordinateOut.y = threadIdx.y+blockIdx.y * blockDim.y;
    coordinateOut.z = threadIdx.z+blockIdx.z * blockDim.z;

    if(coordinateOut<imageOut.getDims())
    {
        double val = 0;
        double kernelFactor = 0;
        Vec<float> inputCenter = Vec<float>(coordinateOut+0.5) / resizeFactors;
        Vec<float> kernelCenter = Vec<float>(kernelDims-1)/2.0f;
        Vec<int> kernelStart(0,0,0);
        Vec<int> kernelEnd(0,0,0);
        
        Vec<float> neighborhoodStart = inputCenter-Vec<float>(kernelDims)/2.0f;
        // if the input start position is negative, we need to start further in on the kernel
        kernelStart.x = (neighborhoodStart.x>=0.0f) ? (0) : (ceil(-neighborhoodStart.x));
        kernelStart.y = (neighborhoodStart.y>=0.0f) ? (0) : (ceil(-neighborhoodStart.y));
        kernelStart.z = (neighborhoodStart.z>=0.0f) ? (0) : (ceil(-neighborhoodStart.z));
        neighborhoodStart = Vec<float>::max(Vec<float>(0.0f, 0.0f, 0.0f), neighborhoodStart);

        // This is the last place to visit in the input (inclusive)
        Vec<float> neighborhoodEnd = inputCenter+(Vec<float>(kernelDims)/2.0f);
        // if the input end position is outside the image, we need to end earlier in on the kernel
        kernelEnd.x = (neighborhoodEnd.x<=imageIn.getDims().x) ? (kernelDims.x) :
            (kernelDims.x-(neighborhoodEnd.x-imageIn.getDims().x));// will floor to int value
        kernelEnd.y = (neighborhoodEnd.y<=imageIn.getDims().y) ? (kernelDims.y) :
            (kernelDims.y-(neighborhoodEnd.y-imageIn.getDims().y));// will floor to int value
        kernelEnd.z = (neighborhoodEnd.z<=imageIn.getDims().z) ? (kernelDims.z) :
            (kernelDims.z-(neighborhoodEnd.z-imageIn.getDims().z));// will floor to int value

        neighborhoodEnd = Vec<float>::min(imageIn.getDims(), neighborhoodEnd);

        Vec<int> curKernelPos(0, 0, 0);
        Vec<int> curInPos = neighborhoodStart;
        for(curKernelPos.z = kernelStart.z; curKernelPos.z<kernelEnd.z; ++curKernelPos.z)
        {
            curInPos.z = neighborhoodStart.z + curKernelPos.z;
            for(curKernelPos.y = kernelStart.y; curKernelPos.y<kernelEnd.y; ++curKernelPos.y)
            {
                curInPos.y = neighborhoodStart.y+curKernelPos.y;
                for(curKernelPos.x = kernelStart.x; curKernelPos.x<kernelEnd.x; ++curKernelPos.x)
                {
                    curInPos.x = neighborhoodStart.x+curKernelPos.x;
                    double imVal = imageIn(curInPos);
                    val += imVal;
                    ++kernelFactor;
                }
            }
        }

        double meanVal = 0;
        if(kernelFactor>0)
        {
            meanVal = val/kernelFactor;
            meanVal = (meanVal>minVal) ? (meanVal) : ((double)minVal);
            meanVal = (meanVal<maxVal) ? (meanVal) : ((double)maxVal);
        }

        imageOut(coordinateOut) = (PixelType)meanVal;
    }
}

template <class PixelType>
PixelType* cResize(const PixelType* imageIn, Vec<size_t> dimsIn, Vec<double> resizeFactors, Vec<size_t>& dimsOut,
                   ReductionMethods method = REDUC_MEAN, PixelType** imageOut = NULL, int device = 0)
{
    cudaSetDevice(device);
    if(resizeFactors.maxValue()<=0)
        resizeFactors = Vec<double>(dimsOut)/Vec<double>(dimsIn);

    dimsOut = Vec<size_t>(Vec<double>(dimsIn)*resizeFactors);

    PixelType* resizedImage = NULL;
    if(imageOut==NULL)
    {
        resizedImage = new PixelType[dimsOut.product()];
        *imageOut = resizedImage;
    }
    else
        resizedImage = *imageOut;

    if(resizeFactors.product()==1)
    {
        memcpy(*imageOut, imageIn, sizeof(PixelType)*dimsIn.product());
        return *imageOut;
    }

    double memSizeRatio = (double)dimsOut.product()/(double)dimsIn.product();
    bool reduce = memSizeRatio<1;

    if(!reduce)
        std::runtime_error("Enlarging is currently not implemented.");

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    size_t memAvail, total;
    cudaMemGetInfo(&memAvail, &total);

    float* hostKernel;
    Vec<size_t> neighborhood;
    Vec<double> neighborhood_ = Vec<double>::max(resizeFactors,Vec<double>(1.0,1.0,1.0)/resizeFactors);

    int blockSize = props.maxThreadsPerBlock;
    if(method==REDUC_GAUS)
    {
        Vec<float> sigmas = Vec<float>(neighborhood_*3);

        neighborhood = createGaussianKernelFull(sigmas, &hostKernel);
        HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*neighborhood.product()));

        //blockSize = getKernelMaxThreads(<PixelType>);
    }
    else if(method==REDUC_MEAN)
    {
        neighborhood = Vec<size_t>(ceil(neighborhood_.x), ceil(neighborhood_.y), ceil(neighborhood_.z));
        hostKernel = new float[neighborhood.product()];
        for(int i = 0; i<neighborhood.product(); ++i)
            hostKernel[i] = 1.0f;

        HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*neighborhood.product()));

        blockSize = getKernelMaxThreads(cudaMeanResize<PixelType>);

        delete[] hostKernel;
    }

    Vec<size_t> bigDims = (reduce) ? (dimsIn) : (dimsOut);
    Vec<size_t> smallDims = (reduce) ? (dimsOut) : (dimsIn);
    memSizeRatio = (reduce) ? (memSizeRatio) : (1/memSizeRatio); // this will be in terms of the smaller image

    std::vector<ImageChunk> bigChunks = calculateBuffers<PixelType>(bigDims, 1, memAvail*MAX_MEM_AVAIL*(1-memSizeRatio), props, neighborhood,blockSize);
    std::vector<ImageChunk> smallChunks(bigChunks);

    for(auto& it:smallChunks)
    {
        Vec<double> start = it.chunkROIstart*resizeFactors;
        it.imageStart = Vec<size_t>(ceil(start.x), ceil(start.y), ceil(start.z));
        it.chunkROIstart = Vec<size_t>(0,0,0);
        it.imageROIstart = it.imageStart;

        Vec<double> end = (it.imageEnd+1)*resizeFactors;
        it.imageEnd = Vec<size_t>(floor(end.x), floor(end.y), floor(end.z))-1;
        it.chunkROIend = it.imageEnd-it.imageStart;
        it.imageROIend = it.imageEnd;
    }

    Vec<size_t> maxBigDeviceDims;
    setMaxDeviceDims(bigChunks, maxBigDeviceDims);
    CudaImageContainerClean<PixelType> deviceBigImage(maxBigDeviceDims, device);

    Vec<size_t> maxSmallDeviceDims;
    setMaxDeviceDims(smallChunks, maxSmallDeviceDims);
    CudaImageContainerClean<PixelType> deviceSmallImage(maxSmallDeviceDims, device);

    std::vector<ImageChunk>::iterator bigIt = bigChunks.begin();
    std::vector<ImageChunk>::iterator smallIt = smallChunks.begin();

    while(bigIt!=bigChunks.end() && smallIt!=smallChunks.end())
    {
        bigIt->sendROI(imageIn, dimsIn, &deviceBigImage);
        deviceSmallImage.setDims(smallIt->getFullChunkSize());

        switch(method)
        {
        case REDUC_MEAN:
            cudaMeanResize<<<smallIt->blocks, smallIt->threads>>>(deviceBigImage, deviceSmallImage, Vec<float>(resizeFactors), neighborhood, std::numeric_limits<PixelType>::min(), std::numeric_limits<PixelType>::max());
            break;
        }
        DEBUG_KERNEL_CHECK();

        smallIt->retriveROI(resizedImage, smallDims, &deviceSmallImage);

        ++bigIt;
        ++smallIt;
    }
    
    return resizedImage;
}
#endif

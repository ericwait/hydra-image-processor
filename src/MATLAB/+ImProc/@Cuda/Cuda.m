classdef (Abstract,Sealed) Cuda
methods (Static)
    commandInfo = Info()
    Help(command)
    imageOut = AddConstant(imageIn,additive,device)
    imageOut = AddImageWith(imageIn1,imageIn2,factor,device)
    imageOut = ApplyPolyTransformation(imageIn,a,b,c,min,max,device)
    imageOut = ContrastEnhancement(imageIn,sigma,MedianNeighborhood,device)
    [numCudaDevices,memoryStats] = DeviceCount()
    deviceStatsArray = DeviceStats()
    imageOut = EntropyFilter(imageIn,kernel,device)
    imageOut = GaussianFilter(imageIn,sigma,device)
    histogram = Histogram(imageIn,numBins,min,max,device)
    imageOut = ImagePow(imageIn,power,device)
    imageOut = LinearUnmixing(mixedImages,unmixMatrix,device)
    imageOut = LoG(imageIn,sigma,device)
    imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations,device)
    imageOut = MaxFilterEllipsoid(imageIn,radius,device)
    imageOut = MaxFilterKernel(imageIn,kernel,device)
    imageOut = MaxFilterNeighborhood(imageIn,Neighborhood,device)
    imageOut = MeanFilter(imageIn,Neighborhood,device)
    imageOut = MedianFilter(imageIn,Neighborhood,device)
    imageOut = MinFilterEllipsoid(imageIn,radius,device)
    imageOut = MinFilterKernel(imageIn,kernel,device)
    imageOut = MinFilterNeighborhood(imageIn,Neighborhood,device)
    [min,max] = MinMax(imageIn,device)
    imageOut = MorphologicalClosure(imageIn,kernel,device)
    imageOut = MorphologicalOpening(imageIn,kernel,device)
    imageOut = MultiplyImage(imageIn,multiplier,device)
    imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor,device)
    normalizedCovariance = NormalizedCovariance(imageIn1,imageIn2,device)
    histogram = NormalizedHistogram(imageIn,numBins,min,max,device)
    imageOut = OtsuThresholdFilter(imageIn,alpha,device)
    threshold = OtsuThresholdValue(imageIn,device)
    maskOut = RegionGrowing(imageIn,kernel,mask,threshold,allowConnections,device)
    imageOut = Resize(imageIn,resizeFactor,explicitSize,method,device)
    sum = SumArray(imageIn,device)
    imageOut = Segment(imageIn,alpha,MorphClosure,device)
    imageOut = StdFilter(imageIn,Neighborhood,device)
    imageOut = ThresholdFilter(imageIn,threshold,device)
    imageOut = TileImage(imageIn,roiStart,roiSize,device)
    variance = Variance(imageIn,device)
    shapeElement = ImProc.MakeBallMask(radius)
end
methods (Static, Access = private)
    varargout = Mex(command, varargin)
end
end

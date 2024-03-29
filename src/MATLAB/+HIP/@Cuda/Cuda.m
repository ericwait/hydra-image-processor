classdef (Abstract,Sealed) Cuda
methods (Static)
    [hydraConfig] = CheckConfig()
    [imageOut] = Closure(imageIn,kernel,numIterations,device)
    [numCudaDevices,memStats] = DeviceCount()
    [deviceStatsArray] = DeviceStats()
    [imageOut] = ElementWiseDifference(image1In,image2In,device)
    [imageOut] = EntropyFilter(imageIn,kernel,device)
    [imageOut] = Gaussian(imageIn,sigmas,numIterations,device)
    [minVal,maxVal] = GetMinMax(imageIn,device)
    Help(command)
    [imageOut] = HighPassFilter(imageIn,sigmas,device)
    [imageOut] = IdentityFilter(imageIn,device)
    [cmdInfo] = Info()
    [imageOut] = LoG(imageIn,sigmas,device)
    [imageOut] = MaxFilter(imageIn,kernel,numIterations,device)
    [imageOut] = MeanFilter(imageIn,kernel,numIterations,device)
    [imageOut] = MedianFilter(imageIn,kernel,numIterations,device)
    [imageOut] = MinFilter(imageIn,kernel,numIterations,device)
    [imageOut] = MultiplySum(imageIn,kernel,numIterations,device)
    [imageOut] = NLMeans(imageIn,h,searchWindowRadius,nhoodRadius,device)
    [imageOut] = Opener(imageIn,kernel,numIterations,device)
    [imageOut] = StdFilter(imageIn,kernel,numIterations,device)
    [imageOut] = Sum(imageIn,device)
    [imageOut] = VarFilter(imageIn,kernel,numIterations,device)
    [imageOut] = WienerFilter(imageIn,kernel,noiseVariance,device)
end
methods (Static, Access = private)
    varargout = HIP(command, varargin)
end
end

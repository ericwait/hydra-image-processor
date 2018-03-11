classdef (Abstract,Sealed) Cuda
methods (Static)
    commandInfo = Info()
    Help(command)
    [numCudaDevices,memoryStats] = DeviceCount()
    deviceStatsArray = DeviceStats()
    arrayOut = MaxFilter(arrayIn,kernel,numIterations,device)
    shapeElement = ImProc.MakeBallMask(radius)
end
methods (Static, Access = private)
    varargout = Mex(command, varargin)
end
end

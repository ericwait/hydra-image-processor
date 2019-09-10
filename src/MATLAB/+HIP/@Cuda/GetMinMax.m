% GetMinMax - This function finds the lowest and highest value in the array that is passed in.
%    [minVal,maxVal] = HIP.Cuda.GetMinMax(imageIn,[device])
%    	imageIn = This is a one to five dimensional array.
%    
%    	device (optional) = Use this if you have multiple devices and want to select one explicitly.
%    		Setting this to [] allows the algorithm to either pick the best device and/or will try to split
%    		the data across multiple devices.
%    
%    	minValue = This is the lowest value found in the array.
%    	maxValue = This is the highest value found in the array.
%    
function [minVal,maxVal] = GetMinMax(imageIn,device)
    [minVal,maxVal] = HIP.Cuda.Mex('GetMinMax',imageIn,device);
end

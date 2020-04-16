% GetMinMax - This function finds the lowest and highest value in the array that is passed in.
%    [minVal,maxVal] = HIP.GetMinMax(imageIn,[device])
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
    try
        [minVal,maxVal] = HIP.Cuda.GetMinMax(imageIn,device);
    catch errMsg
        warning(errMsg.message);
        [minVal,maxVal] = HIP.Local.GetMinMax(imageIn,device);
    end
end

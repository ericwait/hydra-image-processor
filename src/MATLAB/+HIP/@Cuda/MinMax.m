% MinMax - This returns the global min and max values.
%    [minOut,maxOut] = HIP.Cuda.MinMax(arrayIn,[device])
%    	imageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.
%    		The spatial dimensions will have the kernel applied. The last two dimensions will determine
%    		how to stride or jump to the next spatial block.
%    
%    	device (optional) = Use this if you have multiple devices and want to select one explicitly.
%    		Setting this to [] allows the algorithm to either pick the best device and/or will try to split
%    		the data across multiple devices.
%    
%    	minOut = This is the minimum value found in the input.
%    	maxOut = This is the maximum value found in the input.
function [minOut,maxOut] = MinMax(arrayIn,device)
    [minOut,maxOut] = HIP.Cuda.Mex('MinMax',arrayIn,device);
end

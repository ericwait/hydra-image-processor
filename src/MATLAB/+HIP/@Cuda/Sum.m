% Sum - This sums up the entire array in.
%    valueOut = HIP.Cuda.Sum(arrayIn,[device])
%    	imageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.
%    		The spatial dimensions will have the kernel applied. The last two dimensions will determine
%    		how to stride or jump to the next spatial block.
%    
%    	device (optional) = Use this if you have multiple devices and want to select one explicitly.
%    		Setting this to [] allows the algorithm to either pick the best device and/or will try to split
%    		the data across multiple devices.
%    
%    	valueOut = This is the summation of the entire array.
function valueOut = Sum(arrayIn,device)
    [valueOut] = HIP.Cuda.Mex('Sum',arrayIn,device);
end

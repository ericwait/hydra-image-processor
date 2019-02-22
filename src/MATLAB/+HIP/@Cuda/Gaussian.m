% Gaussian - Gaussian smoothing.
%    arrayOut = HIP.Cuda.Gaussian(arrayIn,sigmas,[numIterations],[device])
%    	imageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.
%    		The spatial dimensions will have the kernel applied. The last two dimensions will determine
%    		how to stride or jump to the next spatial block.
%    
%    	Sigmas = This should be an array of three positive values that represent the standard deviation of a Gaussian curve.
%    		Zeros (0) in this array will not smooth in that direction.
%    
%    	numIterations (optional) =  This is the number of iterations to run the max filter for a given position.
%    		This is useful for growing regions by the shape of the structuring element or for very large neighborhoods.
%    		Can be empty an array [].
%    
%    	device (optional) = Use this if you have multiple devices and want to select one explicitly.
%    		Setting this to [] allows the algorithm to either pick the best device and/or will try to split
%    		the data across multiple devices.
%    
%    	imageOut = This will be an array of the same type and shape as the input array.
function arrayOut = Gaussian(arrayIn,sigmas,numIterations,device)
    [arrayOut] = HIP.Cuda.Mex('Gaussian',arrayIn,sigmas,numIterations,device);
end

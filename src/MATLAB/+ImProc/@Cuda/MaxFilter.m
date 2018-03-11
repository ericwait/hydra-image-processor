% MaxFilter - This will set each pixel/voxel to the max value of the neighborhood defined by the given kernel.
%    arrayOut = ImProc.Cuda.MaxFilter(arrayIn,kernel,numIterations,device)
%    	imageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.
%    		The spatial dimensions will have the kernel applied. The last two dimensions will determine
%    		how to stride or jump to the next spatial block.
%    
%    	kernel = This is a one to three dimensional array that will be used to determine neighborhood operations.
%    		In this case, the positions in the kernel that do not equal zeros will be evaluated.
%    		In other words, this can be viewed as a structuring element for the max neighborhood.
%    
%    	numIterations = This is the number of iterations to run the max filter for a given position.
%    		This is useful for growing regions by the shape of the structuring element or for very large neighborhoods.
%    
%    	device = Use this if you have multiple devices and want to select one explicitly. Otherwise set to -1.
%    		Setting this to -1 allows the algorithm to either pick the best device and/or will try to split
%    		the data across multiple devices.
%    
%    	imageOut = This will be an array of the same type and shape as the input array.
function arrayOut = MaxFilter(arrayIn,kernel,numIterations,device)
    [arrayOut] = ImProc.Cuda.Mex('MaxFilter',arrayIn,kernel,numIterations,device);
end

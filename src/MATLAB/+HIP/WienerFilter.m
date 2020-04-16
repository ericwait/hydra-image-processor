% WienerFilter - A Wiener filter aims to denoise an image in a linear fashion.
%    [imageOut] = HIP.WienerFilter(imageIn,kernel,[noiseVariance],[device])
%    	imageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.
%    		The spatial dimensions will have the kernel applied. The last two dimensions will determine
%    		how to stride or jump to the next spatial block.
%    
%    	kernel (optional) = This is a one to three dimensional array that will be used to determine neighborhood operations.
%    		In this case, the positions in the kernel that do not equal zeros will be evaluated.
%    		In other words, this can be viewed as a structuring element for the neighborhood.
%    		 This can be an empty array [] and which will use a 3x3x3 neighborhood (or equivalent given input dimension).
%    
%    	noiseVariance (optional) =  This is the expected variance of the noise.
%    		This should be a scalar value or an empty array [].
%    
%    	device (optional) = Use this if you have multiple devices and want to select one explicitly.
%    		Setting this to [] allows the algorithm to either pick the best device and/or will try to split
%    		the data across multiple devices.
%    
%    	imageOut = This will be an array of the same type and shape as the input array.
%    
function [imageOut] = WienerFilter(imageIn,kernel,noiseVariance,device)
    try
        [imageOut] = HIP.Cuda.WienerFilter(imageIn,kernel,noiseVariance,device);
    catch errMsg
        warning(errMsg.message);
        [imageOut] = HIP.Local.WienerFilter(imageIn,kernel,noiseVariance,device);
    end
end

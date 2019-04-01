% IdentityFilter - Identity Filter for testing. Copies image data to GPU memory and back into output image.
%    [imageOut] = HIP.IdentityFilter(imageIn,[device])
%    	imageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.
%    		The spatial dimensions will have the kernel applied. The last two dimensions will determine
%    		how to stride or jump to the next spatial block.
%    
%    	device (optional) = Use this if you have multiple devices and want to select one explicitly.
%    		Setting this to [] allows the algorithm to either pick the best device and/or will try to split
%    		the data across multiple devices.
%    
%    	imageOut = This will be an array of the same type and shape as the input array.
%    

function [imageOut] = IdentityFilter(imageIn,device)
    try
        [imageOut] = HIP.Cuda.IdentityFilter(imageIn,device);
    catch errMsg
        warning(errMsg.message);
        [imageOut] = HIP.Local.IdentityFilter(imageIn,device);
    end
end

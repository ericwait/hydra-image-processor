% ElementWiseDifference - This subtracts the second array from the first, element by element (A-B).
%    [imageOut] = HIP.ElementWiseDifference(image1In,image2In,[device])
%    	image1In = This is a one to five dimensional array. The first three dimensions are treated as spatial.
%    		The spatial dimensions will have the kernel applied. The last two dimensions will determine
%    		how to stride or jump to the next spatial block.
%    
%    	image2In = This is a one to five dimensional array. The first three dimensions are treated as spatial.
%    		The spatial dimensions will have the kernel applied. The last two dimensions will determine
%    		how to stride or jump to the next spatial block.
%    
%    	device (optional) = Use this if you have multiple devices and want to select one explicitly.
%    		Setting this to [] allows the algorithm to either pick the best device and/or will try to split
%    		the data across multiple devices.
%    
%    	imageOut = This will be an array of the same type and shape as the input array.
function [imageOut] = ElementWiseDifference(image1In,image2In,device)
    try
        [imageOut] = HIP.Cuda.ElementWiseDifference(image1In,image2In,device);
    catch errMsg
        warning(errMsg.message);
        [imageOut] = HIP.Local.ElementWiseDifference(image1In,image2In,device);
    end
end

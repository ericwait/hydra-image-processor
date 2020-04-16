% NLMeans - Apply an approximate non-local means filter using patch mean and covariance with Fisher discrminant distance
%    [imageOut] = HIP.NLMeans(imageIn,h,[searchWindowRadius],[nhoodRadius],[device])
%    	imageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.
%    		The spatial dimensions will have the kernel applied. The last two dimensions will determine
%    		how to stride or jump to the next spatial block.
%    
%    	h = weighting applied to patch difference function. typically e.g. 0.05-0.1. controls the amount of smoothing.
%    
%    	searchWindowRadius = radius of region to locate patches at.
%    
%    	nhoodRadius = radius of patch size (comparison window).
%    
function [imageOut] = NLMeans(imageIn,h,searchWindowRadius,nhoodRadius,device)
    try
        [imageOut] = HIP.Cuda.NLMeans(imageIn,h,searchWindowRadius,nhoodRadius,device);
    catch errMsg
        warning(errMsg.message);
        [imageOut] = HIP.Local.NLMeans(imageIn,h,searchWindowRadius,nhoodRadius,device);
    end
end

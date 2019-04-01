% NLMeans - nonlocal means blahblahblah.
%    [imageOut] = HIP.NLMeans(imageIn,h,[searchWindowRadius],[nhoodRadius],[device])

function [imageOut] = NLMeans(imageIn,h,searchWindowRadius,nhoodRadius,device)
    try
        [imageOut] = HIP.Cuda.NLMeans(imageIn,h,searchWindowRadius,nhoodRadius,device);
    catch errMsg
        warning(errMsg.message);
        [imageOut] = HIP.Local.NLMeans(imageIn,h,searchWindowRadius,nhoodRadius,device);
    end
end

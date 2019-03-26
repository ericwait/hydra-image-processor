% NLMeans - nonlocal means blahblahblah.
%    [imageOut] = HIP.NLMeans(imageIn,a,h,[searchWindowRadius],[nhoodRadius],[device])

function [imageOut] = NLMeans(imageIn,a,h,searchWindowRadius,nhoodRadius,device)
    try
        [imageOut] = HIP.Cuda.NLMeans(imageIn,a,h,searchWindowRadius,nhoodRadius,device);
    catch errMsg
        warning(errMsg.message);
        [imageOut] = HIP.Local.NLMeans(imageIn,a,h,searchWindowRadius,nhoodRadius,device);
    end
end

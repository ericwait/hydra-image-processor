% NLMeans - nonlocal means blahblahblah.
%    [imageOut] = HIP.Cuda.NLMeans(imageIn,h,[searchWindowRadius],[nhoodRadius],[device])
function [imageOut] = NLMeans(imageIn,h,searchWindowRadius,nhoodRadius,device)
    [imageOut] = HIP.Cuda.Mex('NLMeans',imageIn,h,searchWindowRadius,nhoodRadius,device);
end

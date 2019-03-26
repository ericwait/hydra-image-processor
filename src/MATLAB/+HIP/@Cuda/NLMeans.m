% NLMeans - nonlocal means blahblahblah.
%    [imageOut] = HIP.Cuda.NLMeans(imageIn,a,h,[searchWindowRadius],[nhoodRadius],[device])
function [imageOut] = NLMeans(imageIn,a,h,searchWindowRadius,nhoodRadius,device)
    [imageOut] = HIP.Cuda.Mex('NLMeans',imageIn,a,h,searchWindowRadius,nhoodRadius,device);
end

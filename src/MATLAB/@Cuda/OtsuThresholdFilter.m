% OtsuThresholdFilter - imageOut = OtsuThresholdFilter(imageIn,alpha,device) 
function imageOut = OtsuThresholdFilter(imageIn,alpha,device)
    [imageOut] = Cuda.Mex('OtsuThresholdFilter',imageIn,alpha,device);
end

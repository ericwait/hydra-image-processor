% OtsuThresholdFilter - imageOut = OtsuThresholdFilter(imageIn,alpha,device) 
function imageOut = OtsuThresholdFilter(imageIn,alpha,device)
    [imageOut] = ImProc.Cuda.Mex('OtsuThresholdFilter',imageIn,alpha,device);
end

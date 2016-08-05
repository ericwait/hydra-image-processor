% ThresholdFilter - imageOut = ThresholdFilter(imageIn,threshold,device) 
function imageOut = ThresholdFilter(imageIn,threshold,device)
    [imageOut] = ImProc.Cuda.Mex('ThresholdFilter',imageIn,threshold,device);
end

% ThresholdFilter - imageOut = ThresholdFilter(imageIn,threshold,device) 
function imageOut = ThresholdFilter(imageIn,threshold,device)
    [imageOut] = Cuda.Mex('ThresholdFilter',imageIn,threshold,device);
end

% OtsuThresholdValue - threshold = OtsuThresholdValue(imageIn,device) 
function threshold = OtsuThresholdValue(imageIn,device)
    [threshold] = Cuda.Mex('OtsuThresholdValue',imageIn,device);
end

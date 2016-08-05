% OtsuThresholdValue - threshold = OtsuThresholdValue(imageIn,device) 
function threshold = OtsuThresholdValue(imageIn,device)
    [threshold] = ImProc.Cuda.Mex('OtsuThresholdValue',imageIn,device);
end

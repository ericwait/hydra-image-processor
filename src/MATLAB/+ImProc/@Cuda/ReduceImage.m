% ReduceImage - imageOut = ReduceImage(imageIn,reductionFactor,method,device) 
function imageOut = ReduceImage(imageIn,reductionFactor,method,device)
    [imageOut] = ImProc.Cuda.Mex('ReduceImage',imageIn,reductionFactor,method,device);
end

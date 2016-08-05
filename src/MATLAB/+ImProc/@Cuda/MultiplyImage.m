% MultiplyImage - imageOut = MultiplyImage(imageIn,multiplier,device) 
function imageOut = MultiplyImage(imageIn,multiplier,device)
    [imageOut] = ImProc.Cuda.Mex('MultiplyImage',imageIn,multiplier,device);
end

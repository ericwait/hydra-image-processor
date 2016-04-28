% MultiplyImage - imageOut = MultiplyImage(imageIn,multiplier,device) 
function imageOut = MultiplyImage(imageIn,multiplier,device)
    [imageOut] = Cuda.Mex('MultiplyImage',imageIn,multiplier,device);
end

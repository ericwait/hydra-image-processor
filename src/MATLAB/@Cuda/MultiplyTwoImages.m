% MultiplyTwoImages - imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor,device) 
function imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor,device)
    [imageOut] = Cuda.Mex('MultiplyTwoImages',imageIn1,imageIn2,factor,device);
end

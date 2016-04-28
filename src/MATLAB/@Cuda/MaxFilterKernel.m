% MaxFilterKernel - imageOut = MaxFilterKernel(imageIn,kernel,device) 
function imageOut = MaxFilterKernel(imageIn,kernel,device)
    [imageOut] = Cuda.Mex('MaxFilterKernel',imageIn,kernel,device);
end

% MinFilterKernel - imageOut = MinFilterKernel(imageIn,kernel,device) 
function imageOut = MinFilterKernel(imageIn,kernel,device)
    [imageOut] = Cuda.Mex('MinFilterKernel',imageIn,kernel,device);
end

% MinFilterKernel - imageOut = MinFilterKernel(imageIn,kernel,device) 
function imageOut = MinFilterKernel(imageIn,kernel,device)
    [imageOut] = ImProc.Cuda.Mex('MinFilterKernel',imageIn,kernel,device);
end

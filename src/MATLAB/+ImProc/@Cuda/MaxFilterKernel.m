% MaxFilterKernel - imageOut = MaxFilterKernel(imageIn,kernel,device) 
function imageOut = MaxFilterKernel(imageIn,kernel,device)
    [imageOut] = ImProc.Cuda.Mex('MaxFilterKernel',imageIn,kernel,device);
end

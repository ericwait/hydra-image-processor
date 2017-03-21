% EntropyFilter - imageOut = EntropyFilter(imageIn,kernel,device) 
function imageOut = EntropyFilter(imageIn,kernel,device)
    [imageOut] = ImProc.Cuda.Mex('EntropyFilter',imageIn,kernel,device);
end

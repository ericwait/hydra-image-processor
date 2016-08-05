% LinearUnmixing - imageOut = LinearUnmixing(mixedImages,unmixMatrix,device) 
function imageOut = LinearUnmixing(mixedImages,unmixMatrix,device)
    [imageOut] = ImProc.Cuda.Mex('LinearUnmixing',mixedImages,unmixMatrix,device);
end

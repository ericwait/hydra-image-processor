% LinearUnmixing - imageOut = LinearUnmixing(mixedImages,unmixMatrix,device) 
function imageOut = LinearUnmixing(mixedImages,unmixMatrix,device)
    [imageOut] = Cuda.Mex('LinearUnmixing',mixedImages,unmixMatrix,device);
end

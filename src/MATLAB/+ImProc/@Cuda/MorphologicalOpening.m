% MorphologicalOpening - imageOut = MorphologicalOpening(imageIn,kernel,device) 
function imageOut = MorphologicalOpening(imageIn,kernel,device)
    [imageOut] = ImProc.Cuda.Mex('MorphologicalOpening',imageIn,kernel,device);
end

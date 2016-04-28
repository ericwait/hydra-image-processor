% MorphologicalOpening - imageOut = MorphologicalOpening(imageIn,kernel,device) 
function imageOut = MorphologicalOpening(imageIn,kernel,device)
    [imageOut] = Cuda.Mex('MorphologicalOpening',imageIn,kernel,device);
end

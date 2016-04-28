% MorphologicalClosure - imageOut = MorphologicalClosure(imageIn,kernel,device) 
function imageOut = MorphologicalClosure(imageIn,kernel,device)
    [imageOut] = Cuda.Mex('MorphologicalClosure',imageIn,kernel,device);
end

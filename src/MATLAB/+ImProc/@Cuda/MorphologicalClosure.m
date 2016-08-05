% MorphologicalClosure - imageOut = MorphologicalClosure(imageIn,kernel,device) 
function imageOut = MorphologicalClosure(imageIn,kernel,device)
    [imageOut] = ImProc.Cuda.Mex('MorphologicalClosure',imageIn,kernel,device);
end

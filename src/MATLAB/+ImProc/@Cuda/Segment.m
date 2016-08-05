% Segment - imageOut = Segment(imageIn,alpha,MorphClosure,device) 
function imageOut = Segment(imageIn,alpha,MorphClosure,device)
    [imageOut] = ImProc.Cuda.Mex('Segment',imageIn,alpha,MorphClosure,device);
end

% StdFilter - imageOut = StdFilter(imageIn,Neighborhood,device) 
function imageOut = StdFilter(imageIn,Neighborhood,device)
    [imageOut] = ImProc.Cuda.Mex('StdFilter',imageIn,Neighborhood,device);
end

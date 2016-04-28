% StdFilter - imageOut = StdFilter(imageIn,Neighborhood,device) 
function imageOut = StdFilter(imageIn,Neighborhood,device)
    [imageOut] = Cuda.Mex('StdFilter',imageIn,Neighborhood,device);
end

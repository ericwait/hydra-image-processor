% MeanFilter - imageOut = MeanFilter(imageIn,Neighborhood,device) 
function imageOut = MeanFilter(imageIn,Neighborhood,device)
    [imageOut] = Cuda.Mex('MeanFilter',imageIn,Neighborhood,device);
end

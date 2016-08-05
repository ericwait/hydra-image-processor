% MaxFilterNeighborhood - imageOut = MaxFilterNeighborhood(imageIn,Neighborhood,device) 
function imageOut = MaxFilterNeighborhood(imageIn,Neighborhood,device)
    [imageOut] = ImProc.Cuda.Mex('MaxFilterNeighborhood',imageIn,Neighborhood,device);
end

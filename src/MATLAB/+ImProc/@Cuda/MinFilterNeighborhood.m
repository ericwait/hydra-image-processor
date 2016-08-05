% MinFilterNeighborhood - imageOut = MinFilterNeighborhood(imageIn,Neighborhood,device) 
function imageOut = MinFilterNeighborhood(imageIn,Neighborhood,device)
    [imageOut] = ImProc.Cuda.Mex('MinFilterNeighborhood',imageIn,Neighborhood,device);
end

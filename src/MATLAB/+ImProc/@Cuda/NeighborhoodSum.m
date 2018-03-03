% NeighborhoodSum - imageOut = NeighborhoodSum(imageIn,Neighborhood,device) 
function imageOut = NeighborhoodSum(imageIn,Neighborhood,device)
    [imageOut] = ImProc.Cuda.Mex('NeighborhoodSum',imageIn,Neighborhood,device);
end

% RegionGrowing - maskOut = RegionGrowing(imageIn,kernel,mask,threshold,allowConnections,device) 
function maskOut = RegionGrowing(imageIn,kernel,mask,threshold,allowConnections,device)
    [maskOut] = Cuda.Mex('RegionGrowing',imageIn,kernel,mask,threshold,allowConnections,device);
end

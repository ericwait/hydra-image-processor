% RegionGrowing - This will create a mask that grows by a delta value.  If a neighboring voxel is masked and the current voxel intensity is +/- delta from the masked intensity, then the current voxel is added to the mask.
%    maskOut = ImProc.Cuda.RegionGrowing(imageIn,kernel,mask,threshold,allowConnections,device)
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Kernel -- this 
%    
function maskOut = RegionGrowing(imageIn,kernel,mask,threshold,allowConnections,device)
    [maskOut] = ImProc.Cuda.Mex('RegionGrowing',imageIn,kernel,mask,threshold,allowConnections,device);
end

% RegionGrowing - This will create a mask that grows by a delta value.  If a neighboring voxel is masked and the current voxel intensity is +/- delta from the masked intensity, then the current voxel is added to the mask.
%    maskOut = ImProc.RegionGrowing(imageIn,kernel,mask,threshold,allowConnections);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Kernel -- this 
%    
function maskOut = RegionGrowing(imageIn,kernel,mask,threshold,allowConnections)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
       try
            maskOut = ImProc.Cuda.RegionGrowing(imageIn,kernel,mask,threshold,allowConnections,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        maskOut = lclRegionGrowing(imageIn,kernel,mask,threshold,allowConnections);
    end
end

function maskOut = lclRegionGrowing(imageIn,kernel,mask,threshold,allowConnections)

end


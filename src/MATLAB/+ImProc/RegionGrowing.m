% RegionGrowing - This will create a mask that grows by a delta value.  If a neighboring voxel is masked and the current voxel intensity is +/- delta from the masked intensity, then the current voxel is added to the mask.
%    maskOut = ImProc.RegionGrowing(imageIn,kernel,mask,threshold,allowConnections);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Kernel -- this 
%    
function maskOut = RegionGrowing(imageIn,kernel,mask,threshold,allowConnections)
    % check for Cuda capable devices
    curPath = which('ImProc.Cuda');
    curPath = fileparts(curPath);
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       foundDevice = false;
       device = -1;
       
       while(~foundDevice)
        for deviceIdx=1:n
            pause(5*rand(1,1));
            mutexfile = fullfile(curPath,sprintf('device%02d.txt',deviceIdx));
            if (~exist(mutexfile,'file'))
                try
                       fclose(fopen(mutexfile,'wt'));
                catch errMsg
                       continue;
                end
                
                foundDevice = true;
                f = fopen(mutexfile,'at');
                fprintf(f,'%s',devStats(deviceIdx).name);
                fclose(f);
                device = deviceIdx;
                break;
            end
        end
        if (~foundDevice)
            pause(2);
        end
       end
       
       try
            maskOut = ImProc.Cuda.RegionGrowing(imageIn,kernel,mask,threshold,allowConnections,device);
        catch errMsg
        	delete(mutexfile);
        	throw(errMsg);
        end
        
        delete(mutexfile);
    else
        maskOut = lclRegionGrowing(imageIn,kernel,mask,threshold,allowConnections);
    end
end

function maskOut = lclRegionGrowing(imageIn,kernel,mask,threshold,allowConnections)

end


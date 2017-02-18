% MinMax - [min,max] = MinMax(imageIn,device) 
function [min,max] = MinMax(imageIn)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            [min,max] = ImProc.Cuda.MinMax(imageIn,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        [min,max] = lclMinMax(imageIn);
    end
end

function [min,max] = lclMinMax(imageIn)

end


% MinMax - [min,max] = MinMax(imageIn,device) 
function [min,max] = MinMax(imageIn)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
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


% MinMax - [min,max] = MinMax(imageIn,device) 
function [minVal,maxVal] = MinMax(imageIn,forceMATLAB)
    if (~exist('forceMATLAB','var') || isempty(forceMATLAB))
       forceMATLAB = false;
    end
    
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0 && ~forceMATLAB)
       [~,I] = max([m.available]);
       try
            [minVal,maxVal] = ImProc.Cuda.MinMax(imageIn,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        [minVal,maxVal] = lclMinMax(imageIn);
    end
end

function [minVal,maxVal] = lclMinMax(imageIn)
    minVal = min(imageIn(:));
    maxVal = max(imageIn(:));
end


% Variance - variance = Variance(imageIn,device) This will return the variance of an image.
function variance = Variance(imageIn,forceMATLAB)
    if (~exist('forceMATLAB','var') || isempty(forceMATLAB))
       forceMATLAB = false;
    end
    
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0 || ~forceMATLAB)
       [~,I] = max([m.available]);
       try
            variance = ImProc.Cuda.Variance(imageIn,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        variance = lclVariance(imageIn);
    end
end

function variance = lclVariance(imageIn)

end


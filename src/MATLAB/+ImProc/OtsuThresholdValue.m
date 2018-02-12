% OtsuThresholdValue - threshold = OtsuThresholdValue(imageIn,device) 
function threshold = OtsuThresholdValue(imageIn,forceMATLAB)
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
            threshold = ImProc.Cuda.OtsuThresholdValue(imageIn,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        threshold = lclOtsuThresholdValue(imageIn);
    end
end

function threshold = lclOtsuThresholdValue(imageIn)

end


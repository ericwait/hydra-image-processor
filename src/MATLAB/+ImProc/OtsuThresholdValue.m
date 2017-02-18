% OtsuThresholdValue - threshold = OtsuThresholdValue(imageIn,device) 
function threshold = OtsuThresholdValue(imageIn)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
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


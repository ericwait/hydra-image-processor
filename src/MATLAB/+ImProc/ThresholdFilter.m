% ThresholdFilter - imageOut = ThresholdFilter(imageIn,threshold,device) 
function imageOut = ThresholdFilter(imageIn,threshold)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            imageOut = ImProc.Cuda.ThresholdFilter(imageIn,threshold,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclThresholdFilter(imageIn,threshold);
    end
end

function imageOut = lclThresholdFilter(imageIn,threshold)

end


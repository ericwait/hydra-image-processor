% ThresholdFilter - imageOut = ThresholdFilter(imageIn,threshold,device) 
function imageOut = ThresholdFilter(imageIn,threshold)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
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


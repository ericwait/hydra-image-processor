% OtsuThresholdFilter - imageOut = OtsuThresholdFilter(imageIn,alpha,device) 
function imageOut = OtsuThresholdFilter(imageIn,alpha)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            imageOut = ImProc.Cuda.OtsuThresholdFilter(imageIn,alpha,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclOtsuThresholdFilter(imageIn,alpha);
    end
end

function imageOut = lclOtsuThresholdFilter(imageIn,alpha)

end


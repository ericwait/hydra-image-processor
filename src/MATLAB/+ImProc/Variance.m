% Variance - variance = Variance(imageIn,device) This will return the variance of an image.
function variance = Variance(imageIn)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
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


% NormalizedCovariance - normalizedCovariance = NormalizedCovariance(imageIn1,imageIn2,device) 
function normalizedCovariance = NormalizedCovariance(imageIn1,imageIn2)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            normalizedCovariance = ImProc.Cuda.NormalizedCovariance(imageIn1,imageIn2,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        normalizedCovariance = lclNormalizedCovariance(imageIn1,imageIn2);
    end
end

function normalizedCovariance = lclNormalizedCovariance(imageIn1,imageIn2)

end


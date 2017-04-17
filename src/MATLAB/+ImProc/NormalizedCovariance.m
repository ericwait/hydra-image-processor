% NormalizedCovariance - normalizedCovariance = NormalizedCovariance(imageIn1,imageIn2,device) 
function normalizedCovariance = NormalizedCovariance(imageIn1,imageIn2)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
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


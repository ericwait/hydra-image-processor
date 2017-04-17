% OtsuThresholdFilter - imageOut = OtsuThresholdFilter(imageIn,alpha,device) 
function imageOut = OtsuThresholdFilter(imageIn,alpha)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
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


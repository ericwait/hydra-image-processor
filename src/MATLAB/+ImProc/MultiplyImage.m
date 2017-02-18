% MultiplyImage - imageOut = MultiplyImage(imageIn,multiplier,device) 
function imageOut = MultiplyImage(imageIn,multiplier)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            imageOut = ImProc.Cuda.MultiplyImage(imageIn,multiplier,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMultiplyImage(imageIn,multiplier);
    end
end

function imageOut = lclMultiplyImage(imageIn,multiplier)

end


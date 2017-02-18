% LinearUnmixing - imageOut = LinearUnmixing(mixedImages,unmixMatrix,device) 
function imageOut = LinearUnmixing(mixedImages,unmixMatrix)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            imageOut = ImProc.Cuda.LinearUnmixing(mixedImages,unmixMatrix,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclLinearUnmixing(mixedImages,unmixMatrix);
    end
end

function imageOut = lclLinearUnmixing(mixedImages,unmixMatrix)

end


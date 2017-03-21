% EntropyFilter - imageOut = EntropyFilter(imageIn,kernel,device) 
function imageOut = EntropyFilter(imageIn,kernel)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            imageOut = ImProc.Cuda.EntropyFilter(imageIn,kernel,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclEntropyFilter(imageIn,kernel);
    end
end

function imageOut = lclEntropyFilter(imageIn,kernel)

end


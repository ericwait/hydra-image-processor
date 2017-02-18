% SumArray - sum = SumArray(imageIn,device) 
function sum = SumArray(imageIn)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            sum = ImProc.Cuda.SumArray(imageIn,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        sum = lclSumArray(imageIn);
    end
end

function sum = lclSumArray(imageIn)

end


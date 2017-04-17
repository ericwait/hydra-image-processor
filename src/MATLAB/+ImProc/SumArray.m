% SumArray - sum = SumArray(imageIn,device) 
function sum = SumArray(imageIn)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
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


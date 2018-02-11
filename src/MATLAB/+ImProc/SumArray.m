% SumArray - sum = SumArray(imageIn,device) 
function sumVal = SumArray(imageIn,forceMATLAB)
    if (~exist('forceMATLAB','var') || isempty(forceMATLAB))
       forceMATLAB = false;
    end
    
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0 || ~forceMATLAB)
       [~,I] = max([m.available]);
       try
            sumVal = ImProc.Cuda.SumArray(imageIn,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        sumVal = lclSumArray(imageIn);
    end
end

function sumVal = lclSumArray(imageIn)
    sumVal = sum(imageIn(:));
end


% NormalizedHistogram - histogram = NormalizedHistogram(imageIn,numBins,min,max,device) 
function histogram = NormalizedHistogram(imageIn,numBins,min,max)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
       try
            histogram = ImProc.Cuda.NormalizedHistogram(imageIn,numBins,min,max,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        histogram = lclNormalizedHistogram(imageIn,numBins,min,max);
    end
end

function histogram = lclNormalizedHistogram(imageIn,numBins,min,max)

end


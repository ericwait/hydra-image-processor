% NormalizedHistogram - histogram = NormalizedHistogram(imageIn,numBins,min,max,device) 
function histogram = NormalizedHistogram(imageIn,numBins,min,max)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
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


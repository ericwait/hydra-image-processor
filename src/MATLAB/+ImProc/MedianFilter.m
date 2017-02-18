% MedianFilter - imageOut = MedianFilter(imageIn,Neighborhood,device) 
function imageOut = MedianFilter(imageIn,Neighborhood)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            imageOut = ImProc.Cuda.MedianFilter(imageIn,Neighborhood,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMedianFilter(imageIn,Neighborhood);
    end
end

function imageOut = lclMedianFilter(imageIn,Neighborhood)

end


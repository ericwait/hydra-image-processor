% MedianFilter - imageOut = MedianFilter(imageIn,Neighborhood,device) 
function imageOut = MedianFilter(imageIn,Neighborhood)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
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


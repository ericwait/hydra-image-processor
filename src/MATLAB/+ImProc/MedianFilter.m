% MedianFilter - imageOut = MedianFilter(imageIn,Neighborhood,device) 
function imageOut = MedianFilter(imageIn,Neighborhood,forceMATLAB)
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
            imageOut = ImProc.Cuda.MedianFilter(imageIn,Neighborhood,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMedianFilter(imageIn,Neighborhood);
    end
end

function imageOut = lclMedianFilter(imageIn,Neighborhood)
    if (ismatrix(imageIn))
        imageOut = medfilt2(imageIn, Neighborhood([1,2]));
    else
        imageOut = medfilt3(imageIn,Neighborhood);
    end
end


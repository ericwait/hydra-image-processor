% MultiplyImage - imageOut = MultiplyImage(imageIn,multiplier,device) 
function imageOut = MultiplyImage(imageIn,multiplier,forceMATLAB)
    if (~exist('forceMATLAB','var') || isempty(forceMATLAB))
       forceMATLAB = false;
    end
    
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0 && ~forceMATLAB)
       [~,I] = max([m.available]);
       try
            imageOut = ImProc.Cuda.MultiplyImage(imageIn,multiplier,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMultiplyImage(imageIn,multiplier);
    end
end

function imageOut = lclMultiplyImage(imageIn,multiplier)
    imageOut = imageIn .*multiplier;
end


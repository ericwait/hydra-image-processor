% MultiplyTwoImages - imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor,device) 
function imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor,forceMATLAB)
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
            imageOut = ImProc.Cuda.MultiplyTwoImages(imageIn1,imageIn2,factor,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMultiplyTwoImages(imageIn1,imageIn2,factor);
    end
end

function imageOut = lclMultiplyTwoImages(imageIn1,imageIn2,factor)
    imageOut = imageIn1 .* imageIn2 .* factor;
end


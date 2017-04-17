% MultiplyTwoImages - imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor,device) 
function imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
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

end


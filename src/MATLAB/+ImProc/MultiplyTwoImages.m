% MultiplyTwoImages - imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor,device) 
function imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
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


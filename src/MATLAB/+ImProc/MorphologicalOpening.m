% MorphologicalOpening - imageOut = MorphologicalOpening(imageIn,kernel,device) 
function imageOut = MorphologicalOpening(imageIn,kernel)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
       try
            imageOut = ImProc.Cuda.MorphologicalOpening(imageIn,kernel,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMorphologicalOpening(imageIn,kernel);
    end
end

function imageOut = lclMorphologicalOpening(imageIn,kernel)

end


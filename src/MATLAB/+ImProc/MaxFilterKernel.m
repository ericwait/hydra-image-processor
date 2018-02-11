% MaxFilterKernel - imageOut = MaxFilterKernel(imageIn,kernel,device) 
function imageOut = MaxFilterKernel(imageIn,kernel,forceMATLAB)
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
            imageOut = ImProc.Cuda.MaxFilterKernel(imageIn,kernel,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMaxFilterKernel(imageIn,kernel);
    end
end

function imageOut = lclMaxFilterKernel(imageIn,kernel)

end


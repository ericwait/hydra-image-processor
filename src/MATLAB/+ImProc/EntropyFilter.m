% EntropyFilter - imageOut = EntropyFilter(imageIn,kernel,device) 
function imageOut = EntropyFilter(imageIn,kernel,forceMATLAB)
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
            imageOut = ImProc.Cuda.EntropyFilter(imageIn,kernel,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclEntropyFilter(imageIn,kernel);
    end
end

function imageOut = lclEntropyFilter(imageIn,kernel)
    imageOut = entropyfilt(imageIn,size(kernel));
    warning('Falling back to MATLAB version which uses a square nhood and not the kernel passed in.');
end


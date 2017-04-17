% MinFilterKernel - imageOut = MinFilterKernel(imageIn,kernel,device) 
function imageOut = MinFilterKernel(imageIn,kernel)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
       try
            imageOut = ImProc.Cuda.MinFilterKernel(imageIn,kernel,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMinFilterKernel(imageIn,kernel);
    end
end

function imageOut = lclMinFilterKernel(imageIn,kernel)

end


% MarkovRandomFieldDenoiser - imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations,device) 
function imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations,forceMATLAB)
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
            imageOut = ImProc.Cuda.MarkovRandomFieldDenoiser(imageIn,maxIterations,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMarkovRandomFieldDenoiser(imageIn,maxIterations);
    end
end

function imageOut = lclMarkovRandomFieldDenoiser(imageIn,maxIterations)

end


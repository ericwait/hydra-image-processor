% MarkovRandomFieldDenoiser - imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations,device) 
function imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
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


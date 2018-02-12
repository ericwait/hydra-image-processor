% MinFilterNeighborhood - imageOut = MinFilterNeighborhood(imageIn,Neighborhood,device) 
function imageOut = MinFilterNeighborhood(imageIn,Neighborhood,forceMATLAB)
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
            imageOut = ImProc.Cuda.MinFilterNeighborhood(imageIn,Neighborhood,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMinFilterNeighborhood(imageIn,Neighborhood);
    end
end

function imageOut = lclMinFilterNeighborhood(imageIn,Neighborhood)

end


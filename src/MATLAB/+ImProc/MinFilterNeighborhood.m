% MinFilterNeighborhood - imageOut = MinFilterNeighborhood(imageIn,Neighborhood,device) 
function imageOut = MinFilterNeighborhood(imageIn,Neighborhood)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
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


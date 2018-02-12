% MaxFilterEllipsoid - imageOut = MaxFilterEllipsoid(imageIn,radius,device) 
function imageOut = MaxFilterEllipsoid(imageIn,radius,forceMATLAB)
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
            imageOut = ImProc.Cuda.MaxFilterEllipsoid(imageIn,radius,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclMaxFilterEllipsoid(imageIn,radius);
    end
end

function imageOut = lclMaxFilterEllipsoid(imageIn,radius)
    se = ImProc.MakeEllipsoidMask(radius);
    if (ismatrix(imageIn))
        se = se(:,:,round(size(se,3)/2));
    end
    
end


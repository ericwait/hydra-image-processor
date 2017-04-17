% Segment - imageOut = Segment(imageIn,alpha,MorphClosure,device) 
function imageOut = Segment(imageIn,alpha,MorphClosure)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
       try
            imageOut = ImProc.Cuda.Segment(imageIn,alpha,MorphClosure,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclSegment(imageIn,alpha,MorphClosure);
    end
end

function imageOut = lclSegment(imageIn,alpha,MorphClosure)

end


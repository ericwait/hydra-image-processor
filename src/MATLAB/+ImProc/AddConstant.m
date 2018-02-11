% AddConstant - This will add a constant value at every voxel location.
%    imageOut = ImProc.AddConstant(imageIn,additive);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Additive -- must be a double and will be floored if input is an integer type.
%    	ImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.
function imageOut = AddConstant(imageIn,additive,forceMATLAB)
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
            imageOut = ImProc.Cuda.AddConstant(imageIn,additive,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclAddConstant(imageIn,additive);
    end
end

function imageOut = lclAddConstant(imageIn,additive)
    imageOut = imageIn +additive;
end


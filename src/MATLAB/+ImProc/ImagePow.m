% ImagePow - This will raise each voxel value to the power provided.
%    imageOut = ImProc.ImagePow(imageIn,power);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Power -- must be a double.
%    	ImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.
function imageOut = ImagePow(imageIn,power,forceMATLAB)
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
            imageOut = ImProc.Cuda.ImagePow(imageIn,power,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclImagePow(imageIn,power);
    end
end

function imageOut = lclImagePow(imageIn,power)
    imageOut = imageIn .^power;
end


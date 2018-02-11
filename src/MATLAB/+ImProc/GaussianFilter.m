% GaussianFilter - Smooths image using a Gaussian kernel.
%    imageOut = ImProc.GaussianFilter(imageIn,sigma);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Sigma -- these values will create a n-dimensional Gaussian kernel to get a smoothed image that will be subtracted of the original.
%    	ImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.
function imageOut = GaussianFilter(imageIn,sigma,forceMATLAB)
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
            imageOut = ImProc.Cuda.GaussianFilter(imageIn,sigma,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclGaussianFilter(imageIn,sigma);
    end
end

function imageOut = lclGaussianFilter(imageIn,sigma)
    if (ismatrix(imageIn))
        imageOut = imgaussfilt(imageIn,sigma);
    else
        imageOut = imgaussfilt3(imageIn,sigma);
    end
end


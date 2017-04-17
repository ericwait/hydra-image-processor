% ContrastEnhancement - This attempts to increase contrast by removing noise as proposed by Michel et al. This starts with subtracting off a highly smoothed version of imageIn followed by median filter.
%    imageOut = ImProc.ContrastEnhancement(imageIn,sigma,MedianNeighborhood);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Sigma -- these values will create a n-dimensional Gaussian kernel to get a smoothed image that will be subtracted of the original.
%    		N is the number of dimensions of imageIn
%    		The larger the sigma the more object preserving the high pass filter will be (e.g. sigma > 35)
%    	MedianNeighborhood -- this is the neighborhood size in each dimension that will be evaluated for the median neighborhood filter.
%    	ImageOut -- will have the same dimensions and type as imageIn.
function imageOut = ContrastEnhancement(imageIn,sigma,MedianNeighborhood)
    % check for Cuda capable devices
    [devCount,m] = ImProc.Cuda.DeviceCount();
    n = length(devCount);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([m.available]);
       try
            imageOut = ImProc.Cuda.ContrastEnhancement(imageIn,sigma,MedianNeighborhood,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclContrastEnhancement(imageIn,sigma,MedianNeighborhood);
    end
end

function imageOut = lclContrastEnhancement(imageIn,sigma,MedianNeighborhood)

end


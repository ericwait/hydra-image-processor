% ContrastEnhancement - This attempts to increase contrast by removing noise as proposed by Michel et al. This starts with subtracting off a highly smoothed version of imageIn followed by median filter.
%    imageOut = Cuda.ContrastEnhancement(imageIn,sigma,MedianNeighborhood,device)
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Sigma -- these values will create a n-dimensional Gaussian kernel to get a smoothed image that will be subtracted of the original.
%    		N is the number of dimensions of imageIn
%    		The larger the sigma the more object preserving the high pass filter will be (e.g. sigma > 35)
%    	MedianNeighborhood -- this is the neighborhood size in each dimension that will be evaluated for the median neighborhood filter.
%    	Device -- this is an optional parameter that indicates which Cuda capable device to use.
%    	ImageOut -- will have the same dimensions and type as imageIn.
function imageOut = ContrastEnhancement(imageIn,sigma,MedianNeighborhood,device)
    [imageOut] = Cuda.Mex('ContrastEnhancement',imageIn,sigma,MedianNeighborhood,device);
end

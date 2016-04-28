% Histogram - Creates a histogram array with numBins bins between min/max values.
%    histogram = Cuda.Histogram(imageIn,numBins,min,max,device)
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	NumBins -- number of bins that the histogram should partition the signal into.
%    	Min -- this is the minimum value for the histogram.
%    		If min is not provided, the min of the image type is used.
%    	Max -- this is the maximum value for the histogram.
%    		If min is not provided, the min of the image type is used.
%    	Device -- this is an optional parameter that indicates which Cuda capable device to use.
%    	ImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.
function histogram = Histogram(imageIn,numBins,min,max,device)
    [histogram] = Cuda.Mex('Histogram',imageIn,numBins,min,max,device);
end

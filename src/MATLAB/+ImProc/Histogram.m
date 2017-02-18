% Histogram - Creates a histogram array with numBins bins between min/max values.
%    histogram = ImProc.Histogram(imageIn,numBins,min,max);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	NumBins -- number of bins that the histogram should partition the signal into.
%    	Min -- this is the minimum value for the histogram.
%    		If min is not provided, the min of the image type is used.
%    	Max -- this is the maximum value for the histogram.
%    		If min is not provided, the min of the image type is used.
%    	ImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.
function histogram = Histogram(imageIn,numBins,min,max)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            histogram = ImProc.Cuda.Histogram(imageIn,numBins,min,max,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        histogram = lclHistogram(imageIn,numBins,min,max);
    end
end

function histogram = lclHistogram(imageIn,numBins,min,max)

end


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
    curPath = which('ImProc.Cuda');
    curPath = fileparts(curPath);
    n = ImProc.Cuda.DeviceCount();
    foundDevice = false;
    device = -1;
    
    while(~foundDevice)
    	for deviceIdx=1:n
    		mutexfile = fullfile(curPath,sprintf('device%02d.txt',deviceIdx));
    		if (~exist(mutexfile,'file'))
    			try
                    fclose(fopen(mutexfile,'wt'));
    			catch errMsg
                    continue;
    			end
    			foundDevice = true;
    			device = deviceIdx;
    			break;
    		end
    	end
    	if (~foundDevice)
    		pause(2);
    	end
    end
    
    try
        histogram = ImProc.Cuda.Histogram(imageIn,numBins,min,max,device);
    catch errMsg
    	delete(mutexfile);
    	throw(errMsg);
    end
    
    delete(mutexfile);
end

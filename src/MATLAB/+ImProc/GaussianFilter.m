% GaussianFilter - Smooths image using a Gaussian kernel.
%    imageOut = ImProc.GaussianFilter(imageIn,sigma);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Sigma -- these values will create a n-dimensional Gaussian kernel to get a smoothed image that will be subtracted of the original.
%    	ImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.
function imageOut = GaussianFilter(imageIn,sigma)
    curPath = which('ImProc.Cuda');
    curPath = fileparts(curPath);
    n = ImProc.Cuda.DeviceCount();
    foundDevice = false;
    device = -1;
    
    while(~foundDevice)
    	for deviceIdx=1:n
    		mutexfile = fullfile(curPath,sprintf('device%02d.txt',deviceIdx));
    		if (~exist(mutexfile,'file'))
    			f = fopen(mutexfile,'wt');
    			fclose(f);
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
        imageOut = ImProc.Cuda.GaussianFilter(imageIn,sigma,device);
    catch errMsg
    	delete(mutexfile);
    	throw(errMsg);
    end
    
    delete(mutexfile);
end

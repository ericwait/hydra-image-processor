% AddImageWith - This takes two images and adds them together.
%    imageOut = ImProc.AddImageWith(imageIn1,imageIn2,factor);
%    	ImageIn1 -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	ImageIn2 -- can be an image up to three dimensions and of the same type as imageIn1.
%    	Factor -- this is a multiplier to the second image in the form imageOut = imageIn1 + factor*imageIn2.
%    imageOut -- this is the result of imageIn1 + factor*imageIn2 and will be of the same type as imageIn1.
function imageOut = AddImageWith(imageIn1,imageIn2,factor)
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
        imageOut = ImProc.Cuda.AddImageWith(imageIn1,imageIn2,factor,device);
    catch errMsg
    	delete(mutexfile);
    	throw(errMsg);
    end
    
    delete(mutexfile);
end

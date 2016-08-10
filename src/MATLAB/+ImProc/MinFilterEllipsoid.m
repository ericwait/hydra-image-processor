% MinFilterEllipsoid - imageOut = MinFilterEllipsoid(imageIn,radius,device) 
function imageOut = MinFilterEllipsoid(imageIn,radius)
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
        imageOut = ImProc.Cuda.MinFilterEllipsoid(imageIn,radius,device);
    catch errMsg
    	delete(mutexfile);
    	throw(errMsg);
    end
    
    delete(mutexfile);
end

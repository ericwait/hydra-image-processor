% MarkovRandomFieldDenoiser - imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations,device) 
function imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations)
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
        imageOut = ImProc.Cuda.MarkovRandomFieldDenoiser(imageIn,maxIterations,device);
    catch errMsg
    	delete(mutexfile);
    	throw(errMsg);
    end
    
    delete(mutexfile);
end

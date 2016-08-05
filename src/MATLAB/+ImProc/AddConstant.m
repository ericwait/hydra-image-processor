% AddConstant - This will add a constant value at every voxel location.
%    imageOut = ImProc.AddConstant(imageIn,additive);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Additive -- must be a double and will be floored if input is an integer type.
%    	ImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.
function imageOut = AddConstant(imageIn,additive)
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
        imageOut = ImProc.Cuda.AddConstant(imageIn,additive,device);
    catch errMsg
    	delete(mutexfile);
    	throw(errMsg);
    end
    
    delete(mutexfile);
end

% TileImage - This will output an image that only consists of the region of interest indicated.
%    imageOut = ImProc.TileImage(imageIn,roiStart,roiSize);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	RoiStart -- this is the location of the first voxel in the region of interest (starting from the origin).  Must be the same dimension as imageIn.
%    	RoiSize -- this is how many voxels to include starting from roiStart. Must be the same dimension as imageIn.
%    	ImageOut -- this will be an image that only contains the region of interest indicated.
function imageOut = TileImage(imageIn,roiStart,roiSize)
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
        imageOut = ImProc.Cuda.TileImage(imageIn,roiStart,roiSize,device);
    catch errMsg
    	delete(mutexfile);
    	throw(errMsg);
    end
    
    delete(mutexfile);
end

% TileImage - This will output an image that only consists of the region of interest indicated.
%    imageOut = ImProc.TileImage(imageIn,roiStart,roiSize);
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	RoiStart -- this is the location of the first voxel in the region of interest (starting from the origin).  Must be the same dimension as imageIn.
%    	RoiSize -- this is how many voxels to include starting from roiStart. Must be the same dimension as imageIn.
%    	ImageOut -- this will be an image that only contains the region of interest indicated.
function imageOut = TileImage(imageIn,roiStart,roiSize)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            imageOut = ImProc.Cuda.TileImage(imageIn,roiStart,roiSize,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclTileImage(imageIn,roiStart,roiSize);
    end
end

function imageOut = lclTileImage(imageIn,roiStart,roiSize)

end


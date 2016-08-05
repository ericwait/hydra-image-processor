% TileImage - This will output an image that only consists of the region of interest indicated.
%    imageOut = ImProc.Cuda.TileImage(imageIn,roiStart,roiSize,device)
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	RoiStart -- this is the location of the first voxel in the region of interest (starting from the origin).  Must be the same dimension as imageIn.
%    	RoiSize -- this is how many voxels to include starting from roiStart. Must be the same dimension as imageIn.
%    	Device -- this is an optional parameter that indicates which Cuda capable device to use.
%    	ImageOut -- this will be an image that only contains the region of interest indicated.
function imageOut = TileImage(imageIn,roiStart,roiSize,device)
    [imageOut] = ImProc.Cuda.Mex('TileImage',imageIn,roiStart,roiSize,device);
end

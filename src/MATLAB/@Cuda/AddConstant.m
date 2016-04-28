% AddConstant - This will add a constant value at every voxel location.
%    imageOut = Cuda.AddConstant(imageIn,additive,device)
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Additive -- must be a double and will be floored if input is an integer type.
%    	Device -- this is an optional parameter that indicates which Cuda capable device to use.
%    	ImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.
function imageOut = AddConstant(imageIn,additive,device)
    [imageOut] = Cuda.Mex('AddConstant',imageIn,additive,device);
end

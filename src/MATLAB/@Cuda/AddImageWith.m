% AddImageWith - This takes two images and adds them together.
%    imageOut = Cuda.AddImageWith(imageIn1,imageIn2,factor,device)
%    	ImageIn1 -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	ImageIn2 -- can be an image up to three dimensions and of the same type as imageIn1.
%    	Factor -- this is a multiplier to the second image in the form imageOut = imageIn1 + factor*imageIn2.
%    	Device -- this is an optional parameter that indicates which Cuda capable device to use.
%    imageOut -- this is the result of imageIn1 + factor*imageIn2 and will be of the same type as imageIn1.
function imageOut = AddImageWith(imageIn1,imageIn2,factor,device)
    [imageOut] = Cuda.Mex('AddImageWith',imageIn1,imageIn2,factor,device);
end

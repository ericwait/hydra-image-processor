% Resize - Resizes image using various methods.
%    imageOut = ImProc.Cuda.Resize(imageIn,resizeFactor,explicitSize,method,device)
%    	ImageIn -- can be an image up to three dimensions and of type (logical,uint8,int8,uint16,int16,uint32,int32,single,double).
%    	ResizeFactor_rcz -- This represents the output size relative to input (r,c,z). Values less than one but greater than zero will reduce the image.
%    		Values greater than one will enlarge the image. If this is an empty array, it will be calculated from the explicit resize.
%    			If both resizeFactor and explicitSize are both set, the explicitSize will be used.
%    	ExplicitSize_rcz -- This is the size that the output should be (r,c,z). If this is an empty array, then the resize factor is used.
%    			If both resizeFactor and explicitSize are both set, the explicitSize will be used.
%    	Method -- This is the neighborhood operation to apply when resizing (mean, median, min, max, gaussian).
%    	Device -- this is an optional parameter that indicates which Cuda capable device to use.
%    	ImageOut -- This will be a resize image the same type as the input image.
function imageOut = Resize(imageIn,resizeFactor,explicitSize,method,device)
    [imageOut] = ImProc.Cuda.Mex('Resize',imageIn,resizeFactor,explicitSize,method,device);
end

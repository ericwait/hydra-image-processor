% ApplyPolyTransformation - This returns an image with the quadradic function applied. ImageOut = a*ImageIn^2 + b*ImageIn + c
%    imageOut = ImProc.Cuda.ApplyPolyTransformation(imageIn,a,b,c,min,max,device)
%    	A -- this multiplier is applied to the square of the image.
%    	B -- this multiplier is applied to the image.
%    	C -- is the constant additive.
%    	Min -- this is an optional parameter to clamp the output to and is useful for signed or floating point to remove negative values.
%    	Max -- this is an optional parameter to clamp the output to.
%    	Device -- this is an optional parameter that indicates which Cuda capable device to use.
%    	ImageOut -- this is the result of ImageOut = a*ImageIn^2 + b*ImageIn + c and is the same dimension and type as imageIn.
function imageOut = ApplyPolyTransformation(imageIn,a,b,c,min,max,device)
    [imageOut] = ImProc.Cuda.Mex('ApplyPolyTransformation',imageIn,a,b,c,min,max,device);
end

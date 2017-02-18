% ApplyPolyTransformation - This returns an image with the quadradic function applied. ImageOut = a*ImageIn^2 + b*ImageIn + c
%    imageOut = ImProc.ApplyPolyTransformation(imageIn,a,b,c,min,max);
%    	A -- this multiplier is applied to the square of the image.
%    	B -- this multiplier is applied to the image.
%    	C -- is the constant additive.
%    	Min -- this is an optional parameter to clamp the output to and is useful for signed or floating point to remove negative values.
%    	Max -- this is an optional parameter to clamp the output to.
%    	ImageOut -- this is the result of ImageOut = a*ImageIn^2 + b*ImageIn + c and is the same dimension and type as imageIn.
function imageOut = ApplyPolyTransformation(imageIn,a,b,c,min,max)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            imageOut = ImProc.Cuda.ApplyPolyTransformation(imageIn,a,b,c,min,max,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclApplyPolyTransformation(imageIn,a,b,c,min,max);
    end
end

function imageOut = lclApplyPolyTransformation(imageIn,a,b,c,min,max)

end


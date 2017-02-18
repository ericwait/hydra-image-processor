% AddImageWith - This takes two images and adds them together.
%    imageOut = ImProc.AddImageWith(imageIn1,imageIn2,factor);
%    	ImageIn1 -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	ImageIn2 -- can be an image up to three dimensions and of the same type as imageIn1.
%    	Factor -- this is a multiplier to the second image in the form imageOut = imageIn1 + factor*imageIn2.
%    	imageOut -- this is the result of imageIn1 + factor*imageIn2 and will be of the same type as imageIn1.
function imageOut = AddImageWith(imageIn1,imageIn2,factor)
    % check for Cuda capable devices
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       [~,I] = max([devStats.totalMem]);
       try
            imageOut = ImProc.Cuda.AddImageWith(imageIn1,imageIn2,factor,I);
        catch errMsg
        	throw(errMsg);
        end
        
    else
        imageOut = lclAddImageWith(imageIn1,imageIn2,factor);
    end
end

function imageOut = lclAddImageWith(imageIn1,imageIn2,factor)

end


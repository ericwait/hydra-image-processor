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
    curPath = which('ImProc.Cuda');
    curPath = fileparts(curPath);
    devStats = ImProc.Cuda.DeviceStats();
    n = length(devStats);
    
    % if there are devices find the availble one and grab the mutex
    if (n>0)
       foundDevice = false;
       device = -1;
       
       while(~foundDevice)
        for deviceIdx=1:n
            pause(5*rand(1,1));
            mutexfile = fullfile(curPath,sprintf('device%02d.txt',deviceIdx));
            if (~exist(mutexfile,'file'))
                try
                       fclose(fopen(mutexfile,'wt'));
                catch errMsg
                       continue;
                end
                
                foundDevice = true;
                f = fopen(mutexfile,'at');
                fprintf(f,'%s',devStats(deviceIdx).name);
                fclose(f);
                device = deviceIdx;
                break;
            end
        end
        if (~foundDevice)
            pause(2);
        end
       end
       
       try
            imageOut = ImProc.Cuda.ApplyPolyTransformation(imageIn,a,b,c,min,max,device);
        catch errMsg
        	delete(mutexfile);
        	throw(errMsg);
        end
        
        delete(mutexfile);
    else
        imageOut = lclApplyPolyTransformation(imageIn,a,b,c,min,max);
    end
end

function imageOut = lclApplyPolyTransformation(imageIn,a,b,c,min,max)

end


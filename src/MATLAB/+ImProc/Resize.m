% Resize - Resizes image using various methods.
%    imageOut = ImProc.Resize(imageIn,resizeFactor,explicitSize,method);
%    	ImageIn -- can be an image up to three dimensions and of type (logical,uint8,int8,uint16,int16,uint32,int32,single,double).
%    	ResizeFactor_rcz -- This represents the output size relative to input (r,c,z). Values less than one but greater than zero will reduce the image.
%    		Values greater than one will enlarge the image. If this is an empty array, it will be calculated from the explicit resize.
%    			If both resizeFactor and explicitSize are both set, the explicitSize will be used.
%    	ExplicitSize_rcz -- This is the size that the output should be (r,c,z). If this is an empty array, then the resize factor is used.
%    			If both resizeFactor and explicitSize are both set, the explicitSize will be used.
%    	Method -- This is the neighborhood operation to apply when resizing (mean, median, min, max, gaussian).
%    	ImageOut -- This will be a resize image the same type as the input image.
function imageOut = Resize(imageIn,resizeFactor,explicitSize,method)
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
            imageOut = ImProc.Cuda.Resize(imageIn,resizeFactor,explicitSize,method,device);
        catch errMsg
        	delete(mutexfile);
        	throw(errMsg);
        end
        
        delete(mutexfile);
    else
        imageOut = lclResize(imageIn,resizeFactor,explicitSize,method);
    end
end

function imageOut = lclResize(imageIn,resizeFactor,explicitSize,method)

end


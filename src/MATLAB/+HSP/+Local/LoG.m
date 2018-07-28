function arrayOut = LoG(arrayIn,sigmas,device,suppressWarning)
     if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
         warning('Falling back to 2D matlab.');
     end
     
     hsize = sigmas.*10;
     hsize = hsize(1:2);
     sigmas = sigmas(1:2);
     
     h = fspecial('log',hsize,sigmas(1));
     
     arrayOut = arrayIn;
     for t=1:size(arrayIn,5)
         for c=1:size(arrayIn,4)
             for z=1:size(arrayIn,3)
                 arrayOut(:,:,z,c,t) = imfilter(arrayIn(:,:,z,c,t),h);
             end
         end
     end
end

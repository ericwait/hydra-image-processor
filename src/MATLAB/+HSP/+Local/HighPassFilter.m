function arrayOut = HighPassFilter(arrayIn,sigmas,device,suppressWarning)
    if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
        warning('Falling back to matlab.');
    end     
     
     arrayOut = arrayIn;
     for t=1:size(arrayIn,5)
         for c=1:size(arrayIn,4)
             arrayOut(:,:,:,c,t) = imgaussfilt3(arrayIn(:,:,:,c,t),sigmas);
             arrayOut(:,:,:,c,t) = arrayIn(:,:,:,c,t) - arrayOut(:,:,:,c,t);
         end
     end
end

function arrayOut = HighPassFilter(arrayIn,sigmas,device)
     warning('Falling back to matlab.');
     
     arrayOut = arrayIn;
     for t=1:size(im,5)
         for c=1:size(im,4)
             arrayOut(:,:,:,c,t) = imgaussfilt3(arrayIn(:,:,:,c,t),sigmas);
             arrayOut(:,:,:,c,t) = arrayIn(:,:,:,c,t) - arrayOut(:,:,:,c,t);
         end
     end
end

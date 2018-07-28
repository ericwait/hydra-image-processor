function arrayOut = MeanFilter(arrayIn,kernel,numIterations,device,suppressWarning)
    if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
        warning('Falling back to matlab.');
    end
     
     if (~exist('numIterations','var') || isempty(numIterations))
         numIterations = 1;
     end

     arrayOut = arrayIn;
     for t=1:size(arrayIn,5)
         for c=1:size(arrayIn,4)
             for i=1:numIterations
                 if (size(kernel,3)>1 && size(arrayIn,3)>1)
                     arrayOut(:,:,:,c,t) = imboxfilt3(arrayIn(:,:,:,c,t),size(kernel));
                 else
                     for z=1:size(arrayIn,3)
                         arrayOut(:,:,z,c,t) = imboxfilt(arrayIn(:,:,z,c,t),size(kernel(:,:,1)));
                     end
                 end
             end
         end
     end
end

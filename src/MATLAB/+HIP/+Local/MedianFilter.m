function arrayOut = MedianFilter(arrayIn,kernel,numIterations,device,suppressWarning)
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
                     arrayOut(:,:,:,c,t) = medfilt3(arrayIn(:,:,:,c,t),size(kernel));
                 elseif (size(kernel,2)>1 && size(arrayIn,2)>1)
                     for z=1:size(arrayIn,3)
                        arrayOut(:,:,z,c,t) = medfilt2(arrayIn(:,:,z,c,t),[size(kernel,1),size(kernel,2)]);
                     end
                 else
                     for z=1:size(arrayIn,3)
                         for x=1:size(arrayIn,2)
                             arrayOut(:,x,z,c,t) = medfilt1(arrayIn(:,x,z,c,t),size(kernel,1));
                         end
                     end
                 end
             end
         end
     end
end

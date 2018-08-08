function arrayOut = MaxFilter(arrayIn,kernel,numIterations,device,suppressWarning)
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
                 arrayOut(:,:,:,c,t) = imdilate(arrayIn(:,:,:,c,t),kernel);
             end
         end
     end
end

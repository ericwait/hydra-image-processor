function arrayOut = MultiplySum(arrayIn,kernel,numIterations,device)
     warning('Falling back to matlab.');
     
     if (~exist('numIterations','var') || isempty(numIterations))
         numIterations = 1;
     end
     
     arrayOut = arrayIn;
     for t=1:size(arrayIn,5)
         for c=1:size(arrayIn,4)
             for i=1:numIterations
                 arrayOut(:,:,:,c,t) = convn(arrayIn(:,:,:,c,t),kernel,'same');
             end
         end
     end
end

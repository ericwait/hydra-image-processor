function [imageOut] = NLMeans(imageIn,a,h,searchWindowRadius,nhoodRadius,device,suppressWarning)
     error('NLMeans not yet implemented in MATLAB!'); %delete this line when implemented
     if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
         error('Falling back to matlab. But this function is not implemented yet...');
     end
     
     if (~exist('numIterations','var') || isempty(numIterations))
         numIterations = 1;
     end
     
     arrayOut = arrayIn;
     for t=1:size(arrayIn,5)
         for c=1:size(arrayIn,4)
             for i=1:numIterations
                 % implement this function here
                 arrayOut(:,:,:,c,t) = arrayIn(:,:,:,c,t);
             end
         end
     end
end

function arrayOut = EntropyFilter(arrayIn,kernel,device,suppressWarning)
     %error('EntropyFilter not yet implemented in MATLAB!'); %delete this line when implemented
     if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
         warning('Falling back to matlab.');
     end
     
     arrayOut = arrayIn;
     for t=1:size(arrayIn,5)
         for c=1:size(arrayIn,4)
             % implement this function here
             arrayOut(:,:,:,c,t) = entropyfilt(arrayIn(:,:,:,c,t),kernel);
         end
     end
end

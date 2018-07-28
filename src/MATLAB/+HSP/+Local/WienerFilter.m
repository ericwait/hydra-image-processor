function arrayOut = WienerFilter(arrayIn,kernel,noiseVariance,device,suppressWarning)
     if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
         warning('Falling back to 2D matlab.');
     end
     
     if (~exist('kernel','var'))
         kernel = [];
     end
     if (~exist('noiseVariance','var'))
         noiseVariance = [];
     end
     
     arrayOut = arrayIn;
     for t=1:size(arrayIn,5)
         for c=1:size(arrayIn,4)
             for z=1:size(arrayIn,3)
                 arrayOut(:,:,z,c,t) = wiener2(arrayIn(:,:,z,c,t),[size(kernel,1),size(kernel,2)]);
             end
         end
     end
end

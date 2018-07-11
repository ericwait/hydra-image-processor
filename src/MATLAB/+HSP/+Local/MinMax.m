function [minOut,maxOut] = MinMax(arrayIn,device,suppressWarning)
     if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
         warning('Falling back to matlab.');
     end
     
     minOut = min(arrayIn(:));
     maxOut = max(arrayIn(:));
end

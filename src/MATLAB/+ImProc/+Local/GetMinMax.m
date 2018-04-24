function [minValue,maxValue] = GetMinMax(arrayIn,device,suppressWarning)
     if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
         warning('Falling back to matlab.');
     end
     minValue = min(arrayIn(:));
     maxValue = max(arrayIn(:));
end

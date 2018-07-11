function valueOut = Sum(arrayIn,device,suppressWarning)
     if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
         warning('Falling back to matlab.');
     end
     
    valueOut = sum(arrayIn(:));
end

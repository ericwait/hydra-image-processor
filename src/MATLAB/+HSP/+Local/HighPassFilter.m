function arrayOut = HighPassFilter(arrayIn,sigmas,device,suppressWarning)
    if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
        warning('Falling back to matlab.');
    end     
     
    arrayOut = HSP.Local.Gaussian(arrayIn,sigmas,[],[],true);
    arrayOut = arrayOut - arrayIn;
end

function arrayOut = ElementWiseDifference(array1In,array2In,device,suppressWarning)
    if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
        warning('Falling back to matlab.');
    end
    
     arrayOut = array1In-array2In;
end

function arrayOut = Gaussian(arrayIn,sigmas,numIterations,device,suppressWarning)
    if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
        warning('Falling back to matlab.');
    end
    
    if (~exist('numIterations','var') || isempty(numIterations))
        numIterations = 1;
    end
    
    for t=1:size(arrayIn,5)
        for c=1:size(arrayIn,4)
            for i=1:numIterations
                arrayOut(:,:,:,c,t) = imgaussfilt3(arrayIn(:,:,:,c,t),sigmas);
            end
        end
    end
end

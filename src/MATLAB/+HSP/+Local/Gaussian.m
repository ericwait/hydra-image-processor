function arrayOut = Gaussian(arrayIn,sigmas,numIterations,device,suppressWarning)
    if (~exist('suppressWarning','var') || isempty(suppressWarning) || ~suppressWarning)
        warning('Falling back to matlab.');
    end
    
    if (~exist('numIterations','var') || isempty(numIterations))
        numIterations = 1;
    end
    
    arrayOut = zeros(size(arrayIn),'like',arrayIn);
    
    for t=1:size(arrayIn,5)
        for c=1:size(arrayIn,4)
            for i=1:numIterations
                if (sigmas(3)~=0 && size(arrayIn,3)>1)
                    arrayOut(:,:,:,c,t) = imgaussfilt3(arrayIn(:,:,:,c,t),sigmas);
                else
                    for z=1:size(arrayIn,3)
                        arrayOut(:,:,z,c,t) = imgaussfilt(arrayIn(:,:,z,c,t),sigmas(1:2));
                    end
                end
            end
        end
    end
end

function arrayOut = Closure(arrayIn,kernel,numIterations,device) 
    warning('Falling back to matlab and a cuboid region.');
    
    if (~exist('numIterations','var') || isempty(numIterations))
        numIterations = 1;
    end
    
    se = strel('cuboid',size(kernel));
    
    arrayOut = arrayIn;
    for t=1:size(im,5)
        for c=1:size(im,4)
            for i=1:numIterations
                arrayOut(:,:,:,c,t) = imclose(arrayIn(:,:,:,c,t),se);
            end
        end
    end
end

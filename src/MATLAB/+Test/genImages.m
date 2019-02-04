function [imRect, imRand, imSum] = genImages(imDims, pxSize)
    numdims = length(imDims);
    
    genDims = imDims;
    if ( numdims == 1 )
        genDims = [1,imDims];
    end
    
    imRect = zeros(genDims);
    imRand = rand(genDims);
    
    numChans = 1;
    numFrames = 1;
    
    sdims = imDims(1:min(numdims,3));
    
    if ( numdims > 3 )
        numChans = imDims(4);
    end
    if ( numdims > 4 )
        numFrames = imDims(5);
    end
    
    squareSize = 2*(floor(imDims(1)/6));
    
    for t=1:numFrames
        for c=1:numChans
            rectCenter = triRand(sdims/2, sdims/3);
            imRect(:,:,:,c,t) = makeRect(sdims, pxSize, rectCenter, squareSize);
        end
    end
    
    imSum = min(max((imRect + imRand - 0.5), zeros(genDims)), ones(genDims));
end

function imRect = makeRect(imDims, pxSize, rectCenter, diameter)
    if ( length(imDims) == 1 )
        imRect = zeros([1,imDims], 'double');
    else
        imRect = zeros(imDims, 'double');
    end
    
    % Make a "square" assuming non-uniform pixel size
    rectSize = (min(pxSize) ./ pxSize) * diameter;
    
    rectMin = max(round(rectCenter - rectSize/2), ones(1, length(pxSize)));
    rectMax = min(round(rectCenter + rectSize/2), imDims);
    
    rectCoords = arrayfun(@(x,y)(x:y), rectMin,rectMax, 'UniformOutput',false);
    imRect(rectCoords{:}) = 1;
end

function triRnd = triRand(center, width)
    U = 0.5 * (rand(2, length(center)) - 0.5);
    triRnd = width .* sum(U,1) + center;
end

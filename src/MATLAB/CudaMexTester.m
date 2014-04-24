function CudaMexTester(metadataFile,showOut)

totalTime = tic;

device =2;
[~, systemview] = memory;
imageMaxSize = systemview.PhysicalMemory.Available/3 * 0.8;

lowest = 0;
highest= 255;
additive = rand(1)*255;
factor = 1.0;
multiplier = 2.5;
alpha = 1.0;
a = 0.1;
b = 0.5;
c = 0.2;
sigmaX = 30;
sigmaY = 30;
sigmaZ = 10;
NeighborhoodX = 5;
NeighborhoodY = 5;
NeighborhoodZ = 3;
radiusX = 5;
radiusY = 5;
radiusZ = 3;
power = 1.5;
kernel(:,:,1) = [0 1 0; 1 1 1; 0 1 0];
kernel(:,:,2) = [1 1 1; 1 1 1; 1 1 1];
kernel(:,:,3) = [0 1 0; 1 1 1; 0 1 0];

%% run Kernels
for i=1:7
    typeTime = tic;
    switch (i)
        case 1
            typ = 'uint8';
        case 2
            typ = 'uint16';
        case 3
            typ = 'int16';
        case 4
            typ = 'uint32';
        case 5
            typ = 'int32';
        case 6
            typ = 'single';
        case 7
            typ = 'double';
    end
    
    image1 = tiffReader(typ,1,[],[],metadataFile);
    imData = whos('image1');
    redc = imageMaxSize-imData.bytes;
    if (redc<0)
        imTemp = image1(:,:,1);
        imData = whos('imTemp');
        sizeSlice = imData.bytes;
        reduceZ = ceil(-redc/(2*sizeSlice));
        image1 = image1(:,:,reduceZ:end-reduceZ);
        clear imTemp;
    end
    
    image2 = tiffReader(typ,4,[],[],metadataFile);
    imData = whos('image2');
    redc = imageMaxSize-imData.bytes;
    if (redc<0)
        imTemp = image2(:,:,1);
        imData = whos('imTemp');
        sizeSlice = imData.bytes;
        reduceZ = ceil(-redc/(2*sizeSlice));
        image2 = image2(:,:,reduceZ:end-reduceZ);
        clear imTemp;
    end
    
    if (showOut)
        showIm(image1,'Original');
        showIm(image2,'Second Image');
    end
    
    try
        tic
        kernelName = 'AddConstant';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,additive,device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,sprintf('%s of %d',kernelName,additive));
        end
        clear imageOut;
        
        tic
        kernelName = 'AddImageWith';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,image2,factor,device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'ApplyPolyTransformation';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,a,b,c,lowest,highest,device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        MedianNeighborhoodX = NeighborhoodX;
        MedianNeighborhoodY = NeighborhoodY;
        MedianNeighborhoodZ = NeighborhoodZ;
        tic
        kernelName = 'ContrastEnhancement';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[sigmaX,sigmaY,sigmaZ],[MedianNeighborhoodX,MedianNeighborhoodY,MedianNeighborhoodZ],device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'GaussianFilter';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[sigmaX,sigmaY,sigmaZ],device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'Histogram';
        histogram = CudaMex(sprintf('%s',kernelName),image1,255,0,255,device);
        dif = sum(histogram)-length(image1(:));
        fprintf('%s took %f sec and has a dif of %d\n',kernelName,toc,dif);
        if (showOut)
            figure
            plot(1:255,histogram,'-');
            title('Histogram');
        end
        clear imageOut;
        
        tic
        kernelName = 'ImagePow';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,power,device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,sprintf('%s with power %f',kernelName,double(power)));
        end
        clear imageOut;
        
        tic
        kernelName = 'MaxFilterEllipsoid';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[radiusX,radiusY,radiusZ],device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'MaxFilterKernel';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,kernel,device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'MaxFilterNeighborhood';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[NeighborhoodX,NeighborhoodY,NeighborhoodZ],device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'MeanFilter';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[NeighborhoodX,NeighborhoodY,NeighborhoodZ],device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'MedianFilter';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[NeighborhoodX,NeighborhoodY,NeighborhoodZ],device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'MinFilterEllipsoid';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[radiusX,radiusY,radiusZ],device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'MinFilterKernel';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,kernel,device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'MinFilterNeighborhood';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[NeighborhoodX,NeighborhoodY,NeighborhoodZ],device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'MinMax';
        [minVal, maxVal] = CudaMex(sprintf('%s',kernelName),image1,device);
        fprintf('%s took %f sec and returned Min=%f and Max=%f\n',kernelName,toc,minVal,maxVal);
        clear imageOut;
        
        tic
        kernelName = 'MultiplyImage';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,multiplier,device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,sprintf('%s multiplier of %f',kernelName,double(multiplier)));
        end
        clear imageOut;
        
        tic
        kernelName = 'MultiplyTwoImages';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,image2,factor,device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        tic
        kernelName = 'NormalizedHistogram';
        histogram = CudaMex(sprintf('%s',kernelName),image1,255,0,255,device);
        totalVal = sum(histogram);
        fprintf('%s took %f sec and sums to %f\n',kernelName,toc,totalVal);
        if (showOut)
            figure
            plot(1:255,histogram,'-');
            title('Normalized Histogram');
        end
        clear imageOut;
        
        tic
        kernelName = 'OtsuThesholdValue';
        threshold = CudaMex(sprintf('%s',kernelName),image1,device);
        fprintf('%s took %f sec and return a threshold of %f\n',kernelName,toc,double(threshold));
        clear imageOut;
        
        tic
        kernelName = 'OtsuThresholdFilter';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,alpha,device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,kernelName);
        end
        clear imageOut;
        
        reductionFactorX = NeighborhoodX;
        reductionFactorY = NeighborhoodY;
        reductionFactorZ = NeighborhoodZ;
        tic
        kernelName = 'ReduceImage';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[reductionFactorX,reductionFactorY,reductionFactorZ],'mean',device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,[kernelName ' Mean']);
        end
        clear imageOut;
        
        tic
        kernelName = 'ReduceImage';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[reductionFactorX,reductionFactorY,reductionFactorZ],'median',device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,[kernelName ' Median']);
        end
        clear imageOut;
        
        tic
        kernelName = 'ReduceImage';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[reductionFactorX,reductionFactorY,reductionFactorZ],'min',device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,[kernelName ' Min']);
        end
        clear imageOut;
        
        tic
        kernelName = 'ReduceImage';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,[reductionFactorX,reductionFactorY,reductionFactorZ],'max',device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,[kernelName ' Max']);
        end
        clear imageOut;
        
        tic
        kernelName = 'SumArray';
        sumVal = CudaMex(sprintf('%s',kernelName),image1,device);
        dif = sumVal - sum(image1(:));
        fprintf('%s took %f sec and returned a dif of %f\n',kernelName,toc,dif);
        clear imageOut;
        
        threshold = additive;
        tic
        kernelName = 'ThresholdFilter';
        imageOut = CudaMex(sprintf('%s',kernelName),image1,threshold,device);
        fprintf('%s took %f sec\n',kernelName,toc);
        if (showOut)
            showIm(imageOut,sprintf('%s with threshold %d',kernelName,threshold));
        end
        clear imageOut;
        
        zChunk = ceil(size(image1,3)/3);
        zRange = zChunk:zChunk+zChunk;
        
        image1 = image1(:,:,zRange);
        image2 = image2(:,:,zRange);
        tic
        kernelName = 'NormalizedCovariance';
        normalizedCovariance = CudaMex(sprintf('%s',kernelName),image1,image2,device);
        fprintf('%s took %f sec and returned %f\n',kernelName,toc,normalizedCovariance);
        clear imageOut;
    catch e
        fprintf('************\nError: %sFrom line %d\n************\n',e.message,e.stack.line);
        clear mex
        return
    end
    
    fprintf('\n%s took %f min total\n\n',typ,toc(typeTime)/60.0);
end

clear mex

fprintf('\n\nEntire test took a total of %f min\n\n',toc(totalTime)/60);
end



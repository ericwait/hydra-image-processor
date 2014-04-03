function CudaMexTester(imageIn,device)

%% set optional paramaters
imageRand = uint8(rand(size(imageIn))*255);
imageIn1 = imageIn;
min = 0;
max = 255;
additive = rand(1)*255;
factor = 1.0;
multiplier = 2.5;
alpha = 1.0;
a = 0.0;
b = 0.5;
c = 0.2;
sigmaX = 30;
sigmaY = 30;
sigmaZ = 10;
NeighborhoodX = 5;
NeighborhoodY = 5;
NeighborhoodZ = 5;
radiusX = 5;
radiusY = 5;
radiusZ = 3;
power = 1.5;
kernel = ...
    [[0 1 0;...
    1 1 1;...
    0 1 0];...
    [1 1 1;...
    1 1 1;...
    1 1 1];...
    [0 1 0;...
    1 1 1;...
    0 1 0]];

%% run Kernels

showIm(imageIn,'Original');

try
    tic
    kernelName = 'AddConstant';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,additive,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,sprintf('%s of %d',kernelName,additive));
    
    tic
    kernelName = 'AddImageWith';
    imageIn2 = imageRand;
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn1,imageIn2,factor,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'ApplyPolyTransformation';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,a,b,c,min,max,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    MedianNeighborhoodX = NeighborhoodX;
    MedianNeighborhoodY = NeighborhoodY;
    MedianNeighborhoodZ = NeighborhoodZ;
    tic
    kernelName = 'ContrastEnhancement';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,[sigmaX,sigmaY,sigmaZ],[MedianNeighborhoodX,MedianNeighborhoodY,MedianNeighborhoodZ],device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'GaussianFilter';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,[sigmaX,sigmaY,sigmaZ],device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'Histogram';
    histogram = CudaMex(sprintf('%s',kernelName),imageIn,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    figure
    plot(1:255,histogram,'-');
    title('Histogram');
    
    tic
    kernelName = 'ImagePow';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,power,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,sprintf('%s with power %f',kernelName,double(power)));
    
    tic
    kernelName = 'MaxFilterEllipsoid';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,[radiusX,radiusY,radiusZ],device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'MaxFilterKernel';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,kernel,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'MaxFilterNeighborhood';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,[NeighborhoodX,NeighborhoodY,NeighborhoodZ],device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'MeanFilter';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,[NeighborhoodX,NeighborhoodY,NeighborhoodZ],device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'MedianFilter';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,[NeighborhoodX,NeighborhoodY,NeighborhoodZ],device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'MaxFilterEllipsoid';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,[radiusX,radiusY,radiusZ],device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'MinFilterKernel';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,kernel,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'MinFilterNeighborhood';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,[NeighborhoodX,NeighborhoodY,NeighborhoodZ],device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'MultiplyImage';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,multiplier,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,sprintf('%s multiplier of %f',kernelName,double(multiplier)));
    
    tic
    kernelName = 'MultiplyTwoImages';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn1,imageIn2,factor,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'NormalizedCovariance';
    normalizedCovariance = CudaMex(sprintf('%s',kernelName),imageIn1,imageIn2,device);
    fprintf('%s took %f sec and returned %f\n',kernelName,toc,normalizedCovariance);
    
    tic
    kernelName = 'NormalizedHistogram';
    histogram = CudaMex(sprintf('%s',kernelName),imageIn,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    figure
    plot(1:255,histogram,'-');
    title('Normalized Histogram');
    
    tic
    kernelName = 'OtsuThesholdValue';
    threshold = CudaMex(sprintf('%s',kernelName),imageIn,device);
    fprintf('%s took %f sec and return a threshold of %f\n',kernelName,toc,double(threshold));
    
    tic
    kernelName = 'OtsuThresholdFilter';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,alpha,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    reductionFactorX = NeighborhoodX;
    reductionFactorY = NeighborhoodY;
    reductionFactorZ = NeighborhoodZ;
    tic
    kernelName = 'ReduceImage';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,[reductionFactorX,reductionFactorY,reductionFactorZ],device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,kernelName);
    
    tic
    kernelName = 'SumArray';
    sum = CudaMex(sprintf('%s',kernelName),imageIn,device);
    fprintf('%s took %f sec and returned %f\n',kernelName,toc,sum);
    
    threshold = additive;
    tic
    kernelName = 'ThresholdFilter';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,threshold,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,sprintf('%s with threshold %d',kernelName,threshold));
catch e
    fprintf('************\nError: %sFrom line %d\n************\n',e.message,e.stack.line);
    clear mex
    return
end

clear mex
end

%% print out image
function showIm(image,label)
figure
imagesc(max(image,[],3))
colormap gray

title(label)
end

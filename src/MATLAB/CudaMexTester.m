function CudaMexTester(imageIn,device,image2)

%% set optional paramaters
if (~exist('image2','var') || isempty(image2))
    image2 = uint8(rand(size(imageIn))*255);
end

imageIn1 = imageIn;
imageIn2 = image2;
min = 0;
max = 255;
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

showIm(imageIn,'Original');
showIm(image2,'Second Image');

try
    tic
    kernelName = 'AddConstant';
    imageOut = CudaMex(sprintf('%s',kernelName),imageIn,additive,device);
    fprintf('%s took %f sec\n',kernelName,toc);
    showIm(imageOut,sprintf('%s of %d',kernelName,additive));
    
    tic
    kernelName = 'AddImageWith';
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
    histogram = CudaMex(sprintf('%s',kernelName),imageIn,255,0,255,device);
    dif = sum(histogram)-length(imageIn(:));
    fprintf('%s took %f sec and has a dif of %d\n',kernelName,toc,dif);
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
    kernelName = 'MinFilterEllipsoid';
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
    kernelName = 'MinMax';
    [minVal, maxVal] = CudaMex(sprintf('%s',kernelName),imageIn,device);
    fprintf('%s took %f sec and returned Min=%f and Max=%f\n',kernelName,toc,minVal,maxVal);
    
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
    histogram = CudaMex(sprintf('%s',kernelName),imageIn,255,0,255,device);
    totalVal = sum(histogram);
    fprintf('%s took %f sec and sums to %f\n',kernelName,toc,totalVal);
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
    sumVal = CudaMex(sprintf('%s',kernelName),imageIn,device);
    dif = sumVal - sum(imageIn(:));
    fprintf('%s took %f sec and returned a dif of %f\n',kernelName,toc,dif);
    
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



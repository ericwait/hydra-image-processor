function [cTime,mTime,kernelName] = GaussianFilter(im,sigma)
    kernelName = 'GaussianFilter';
    cT = tic;
    imC = ImProc.GaussianFilter(im,sigma);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.GaussianFilter(im,sigma,true);
    mTime = toc(mT);
end

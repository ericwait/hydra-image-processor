function [cTime,mTime,kernelName] = GaussianFilter(im,sigma)
    kernelName = 'GaussianFilter';
    cT = tic;
    imC = ImProc.Gaussian(im,sigma,[],[]);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.Local.Gaussian(im,sigma,[],[],true);
    mTime = toc(mT);
end

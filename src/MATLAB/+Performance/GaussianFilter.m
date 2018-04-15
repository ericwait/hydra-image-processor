function [cTime,mTime,kernelName,c2Time] = GaussianFilter(im,sigma,numDevices)
    kernelName = 'GaussianFilter';
    
    cT = tic;
    imC = ImProc.Gaussian(im,sigma,[],1);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.Local.Gaussian(im,sigma,[],[],true);
    mTime = toc(mT);
    
    if (numDevices>1)
        c2T = tic;
        imC = ImProc.Gaussian(im,sigma,[],[]);
        c2Time = toc(c2T);
    else
        c2Time = inf;
    end
end

function [cTime,mTime,kernelName,c2Time] = GaussianFilter(im,sigma,numDevices)
    kernelName = 'GaussianFilter';
    
    cT = tic;
    imC = HIP.Gaussian(im,sigma,[],1);
    cTime = toc(cT);

    mT = tic;
    imM = HIP.Local.Gaussian(im,sigma,[],[],true);
    mTime = toc(mT);
    
    if (numDevices>1)
        c2T = tic;
        imC = HIP.Gaussian(im,sigma,[],[]);
        c2Time = toc(c2T);
    else
        c2Time = inf;
    end
end

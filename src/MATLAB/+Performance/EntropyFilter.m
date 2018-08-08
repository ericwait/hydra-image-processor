function [cTime,mTime,kernelName] = EntropyFilter(im,nhood)
    kernelName = 'EntropyFilter';
    cT = tic;
    imC = HIP.EntropyFilter(im,ones(nhood));
    cTime = toc(cT);

    mT = tic;
    imM = HIP.EntropyFilter(im,ones(nhood),true);
    mTime = toc(mT);
end

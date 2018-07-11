function [cTime,mTime,kernelName] = EntropyFilter(im,nhood)
    kernelName = 'EntropyFilter';
    cT = tic;
    imC = HSP.EntropyFilter(im,ones(nhood));
    cTime = toc(cT);

    mT = tic;
    imM = HSP.EntropyFilter(im,ones(nhood),true);
    mTime = toc(mT);
end

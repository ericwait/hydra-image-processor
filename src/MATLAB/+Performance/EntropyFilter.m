function [cTime,mTime,kernelName] = EntropyFilter(im,nhood)
    kernelName = 'EntropyFilter';
    cT = tic;
    imC = ImProc.EntropyFilter(im,ones(nhood));
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.EntropyFilter(im,ones(nhood),true);
    mTime = toc(mT);
end
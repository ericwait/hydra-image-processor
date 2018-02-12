function [cTime,mTime,kernelName] = StdFilter(im,nhood)
    kernelName = 'StdFilter';
    cT = tic;
    imC = ImProc.StdFilter(im,nhood);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.StdFilter(im,nhood,true);
    mTime = toc(mT);
end
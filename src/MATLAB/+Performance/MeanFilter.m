function [cTime,mTime,kernelName] = MeanFilter(im,nhood)
    kernelName = 'MeanFilter';
    cT = tic;
    imC = ImProc.MeanFilter(im,nhood,[],[]);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.Local.MeanFilter(im,nhood,[],[],true);
    mTime = toc(mT);
end

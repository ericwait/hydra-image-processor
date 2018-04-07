function [cTime,mTime,kernelName] = MedianFilter(im,nhood)
    kernelName = 'MedianFilter';
    cT = tic;
    imC = ImProc.MedianFilter(im,nhood,[],[]);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.Local.MedianFilter(im,nhood,[],[],true);
    mTime = toc(mT);
end

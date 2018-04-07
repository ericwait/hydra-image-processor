function [cTime,mTime,kernelName] = HighPassFilter(im,gauss)
    kernelName = 'Contrast Enhancement';
    cT = tic;
    imC = ImProc.HighPassFilter(im,gauss,[]);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.Local.HighPassFilter(im,gauss,[],true);
    mTime = toc(mT);
end
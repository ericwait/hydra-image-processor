function [cTime,mTime,kernelName] = SumArray(im)
    kernelName = 'SumArray';
    cT = tic;
    imC = ImProc.SumArray(im);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.SumArray(im,true);
    mTime = toc(mT);
end
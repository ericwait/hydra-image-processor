function [cTime,mTime,kernelName] = SumArray(im)
    kernelName = 'SumArray';
    cT = tic;
    imC = HSP.SumArray(im);
    cTime = toc(cT);

    mT = tic;
    imM = HSP.SumArray(im,true);
    mTime = toc(mT);
end

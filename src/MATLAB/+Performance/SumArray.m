function [cTime,mTime,kernelName] = SumArray(im)
    kernelName = 'SumArray';
    cT = tic;
    imC = HIP.SumArray(im);
    cTime = toc(cT);

    mT = tic;
    imM = HIP.SumArray(im,true);
    mTime = toc(mT);
end

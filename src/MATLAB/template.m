function [cTime,mTime,kernelName] = (im,)
    kernelName = '';
    cT = tic;
    imC = HIP.(im,);
    cTime = toc(cT);

    mT = tic;
    imM = HIP.(im,,true);
    mTime = toc(mT);
end

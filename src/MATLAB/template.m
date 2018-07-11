function [cTime,mTime,kernelName] = (im,)
    kernelName = '';
    cT = tic;
    imC = HSP.(im,);
    cTime = toc(cT);

    mT = tic;
    imM = HSP.(im,,true);
    mTime = toc(mT);
end

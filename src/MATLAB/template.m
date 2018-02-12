function [cTime,mTime,kernelName] = (im,)
    kernelName = '';
    cT = tic;
    imC = ImProc.(im,);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.(im,,true);
    mTime = toc(mT);
end
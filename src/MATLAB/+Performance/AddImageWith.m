function [cTime,mTime,kernelName] = AddImageWith(im)
    kernelName = 'AddImageWith';
    cT = tic;
    imC = ImProc.AddImageWith(im,im,2);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.AddImageWith(im,im,2,true);
    mTime = toc(mT);
end
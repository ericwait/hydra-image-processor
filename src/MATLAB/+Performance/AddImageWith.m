function [cTime,mTime,kernelName] = AddImageWith(im)
    kernelName = 'AddImageWith';
    cT = tic;
    imC = HIP.AddImageWith(im,im,2);
    cTime = toc(cT);

    mT = tic;
    imM = HIP.AddImageWith(im,im,2,true);
    mTime = toc(mT);
end

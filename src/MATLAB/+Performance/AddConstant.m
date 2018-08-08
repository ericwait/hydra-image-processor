function [cTime,mTime,kernelName] = AddConstant(im,additive)
    kernelName = 'AddConstant';
    cT = tic;
    imC = HIP.AddConstant(im,additive);
    cTime = toc(cT);

    mT = tic;
    imM = HIP.AddConstant(im,additive,true);
    mTime = toc(mT);
end

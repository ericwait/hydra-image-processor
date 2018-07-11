function [cTime,mTime,kernelName] = AddConstant(im,additive)
    kernelName = 'AddConstant';
    cT = tic;
    imC = HSP.AddConstant(im,additive);
    cTime = toc(cT);

    mT = tic;
    imM = HSP.AddConstant(im,additive,true);
    mTime = toc(mT);
end

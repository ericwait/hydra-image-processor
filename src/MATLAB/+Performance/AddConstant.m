function [cTime,mTime,kernelName] = AddConstant(im,additive)
    kernelName = 'AddConstant';
    cT = tic;
    imC = ImProc.AddConstant(im,additive);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.AddConstant(im,additive,true);
    mTime = toc(mT);
end
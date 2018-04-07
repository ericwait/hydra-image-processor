function [cTime,mTime,kernelName] = Closure(im,kernel)
    kernelName = 'Closure';
    cT = tic;
    imC = ImProc.Closure(im,kernel,[],[]);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.Local.Closure(im,kernel,[],[],true);
    mTime = toc(mT);
end

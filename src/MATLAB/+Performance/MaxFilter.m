function [cTime,mTime,kernelName] = MaxFilter(im,kernel)
    kernelName = 'MaxFilter';
    cT = tic;
    imC = ImProc.MaxFilter(im,kernel,[],[]);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.Local.MaxFilter(im,kernel,[],[],true);
    mTime = toc(mT);
end

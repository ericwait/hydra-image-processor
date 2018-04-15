function [cTime,mTime,kernelName,c2Time] = HighPassFilter(im,gauss,numDevices)
    kernelName = 'Contrast Enhancement';
    
    cT = tic;
    imC = ImProc.HighPassFilter(im,gauss,1);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.Local.HighPassFilter(im,gauss,[],true);
    mTime = toc(mT);

    if (numDevices>1)
        c2T = tic;
        imC = ImProc.HighPassFilter(im,gauss,[]);
        c2Time = toc(c2T);
    else
        c2Time = inf;
    end
end
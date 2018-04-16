function [cTime,mTime,kernelName,c2Time] = MaxFilter(im,kernel,numDevices)
    kernelName = 'MaxFilter';
    
    cT = tic;
    imC = ImProc.MaxFilter(im,kernel,[],1);
    cTime = toc(cT);
    clear imC

    mT = tic;
    imM = ImProc.Local.MaxFilter(im,kernel,[],[],true);
    mTime = toc(mT);
    clear imM
    
    if (numDevices>1)
        c2T = tic;
        imC = ImProc.MaxFilter(im,kernel,[],[]);
        c2Time = toc(c2T);
    else
        c2Time = inf;
    end
end

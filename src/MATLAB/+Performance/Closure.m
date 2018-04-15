function [cTime,mTime,kernelName,c2Time] = Closure(im,kernel,numDevices)
    kernelName = 'Closure';
    
    cT = tic;
    imC = ImProc.Closure(im,kernel,[],1);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.Local.Closure(im,kernel,[],[],true);
    mTime = toc(mT);
    
    if (numDevices>1)
        c2T = tic;
        imC = ImProc.Closure(im,kernel,[],[]);
        c2Time = toc(c2T);
    else
        c2Time = inf;
    end
end

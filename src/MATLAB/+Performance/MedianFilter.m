function [cTime,mTime,kernelName,c2Time] = MedianFilter(im,nhood,numDevices)
    kernelName = 'MedianFilter';
    
    cT = tic;
    imC = HSP.MedianFilter(im,nhood,[],1);
    cTime = toc(cT);
    
    mT = tic;
    imM = HSP.Local.MedianFilter(im,nhood,[],[],true);
    mTime = toc(mT);
    
    if (numDevices>1)
        c2T = tic;
        imC = HSP.MedianFilter(im,nhood,[],[]);
        c2Time = toc(c2T);
    else
        c2Time = inf;
    end
end

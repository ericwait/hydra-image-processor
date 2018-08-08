function [cTime,mTime,kernelName,c2Time] = MedianFilter(im,nhood,numDevices)
    kernelName = 'MedianFilter';
    
    cT = tic;
    imC = HIP.MedianFilter(im,nhood,[],1);
    cTime = toc(cT);
    
    mT = tic;
    imM = HIP.Local.MedianFilter(im,nhood,[],[],true);
    mTime = toc(mT);
    
    if (numDevices>1)
        c2T = tic;
        imC = HIP.MedianFilter(im,nhood,[],[]);
        c2Time = toc(c2T);
    else
        c2Time = inf;
    end
end

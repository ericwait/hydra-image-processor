function [cTime,mTime,kernelName,c2Time] = MeanFilter(im,nhood,numDevices)
    kernelName = 'MeanFilter';
    
    cT = tic;
    imC = HSP.MeanFilter(im,nhood,[],1);
    cTime = toc(cT);

    mT = tic;
    imM = HSP.Local.MeanFilter(im,nhood,[],[],true);
    mTime = toc(mT);
    
    if (numDevices>1)
        c2T = tic;
        imC = HSP.MeanFilter(im,nhood,[],[]);
        c2Time = toc(c2T);
    else
        c2Time = inf;
    end
end

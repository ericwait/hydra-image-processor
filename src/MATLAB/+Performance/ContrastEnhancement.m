function [cTime,mTime,kernelName] = ContrastEnhancement(im,gauss,med)
    kernelName = 'Contrast Enhancement';
    cT = tic;
    imC = ImProc.ContrastEnhancement(im,gauss,med);
    cTime = toc(cT);

    mT = tic;
    imM = ImProc.ContrastEnhancement(im,gauss,med,true);
    mTime = toc(mT);
end
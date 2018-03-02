function ContrastEnhancement(imarisAppID)
    imarisHandle = ImarisHelper.GetAppHandle(imarisAppID);
    if (isempty(imarisHandle))
        error('Imaris is not open or the wrong app id!');
    end
    
    imarisDataset = imarisHandle.GetDataSet();
    physicalSize = ImarisHelper.GetPhysicalSize(imarisDataset);
    normSize = physicalSize./max(physicalSize);
    defaultSigs = 35.*normSize;
    defaultSigsStr = sprintf('[%.2f,%.2f,%.2f]',defaultSigs(1),defaultSigs(2),defaultSigs(3));
    
    prompts = {'Enter Channel:','Enter Gaussian Sigmas:','Enter Median Neighborhood:','Inplace:(false=0,true=1)'};
    dlgTitle = 'Contrast Enhancement';
    numLines = 1;
    defaultAns = {'1',defaultSigsStr,'[3,3,3]','0'};
    answer = inputdlg(prompts,dlgTitle,numLines,defaultAns);
    
    if (isempty(answer))
        return
    end
    
    chan = str2double(answer{1});
    sigsStr = regexp(answer{2},'\[(.*),(.*),(.*)\]','tokens');
    sigs = abs(cellfun(@(x)(str2double(x)),sigsStr{1}));
    medStr = regexp(answer{3},'\[(.*),(.*),(.*)\]','tokens');
    medNeighborhood = cellfun(@(x)(str2double(x)),medStr{1});
    medNeighborhood = abs(round(medNeighborhood));
    
    inplace = str2double(answer{4})>0;
    outChannel = ImarisHelper.GetNumChannels(imarisDataset) +1;
    if (inplace)
        outChannel = chan;
    end

    imarisHandle.DataSetPushUndo('Contrast Enhancement');
    
    % TODO see if there is enough memory to just capture the entire series
    % to process faster
    
    numFrames = ImarisHelper.GetNumFrames(imarisDataset);
    im = ImarisHelper.GetImageData(imarisDataset,chan,1:numFrames);
    prgs = Utils.CmdlnProgress(numFrames,true,'smooth');
    for t=1:numFrames
        im(:,:,:,1,t) = ImProc.ContrastEnhancement(im(:,:,:,1,t),sigs,medNeighborhood);
        prgs.PrintProgress(t);
    end
    prgs.ClearProgress(true);
    imarisDataset = ImarisHelper.SetImage(imarisHandle,imarisDataset,im,outChannel,1:t);
end

function captureSnapshot(rootDir, prefix)
    tic();
    
    testDir = fullfile(rootDir,'Testing');
    
    if ( ~exist('prefix','var') )
        prefix = 'snapshot';
    end
    
    %% Build a folder name from architecture and git commit (or date)
    archStr = computer('arch');
    [status,revStr] = system(['git -C "' rootDir '" rev-parse HEAD']);
    revStr = strtrim(revStr);
    if ( status ~= 0 )
        revStr = datestr(now(),'yyyy-mm-dd_HHMM');
    end
    
    snapName = [prefix '_matlab_' archStr '_' revStr];
    snapDir = fullfile(testDir, snapName);
    
    if ( ~exist(snapDir,'dir') )
        mkdir(snapDir);
    else
        delete(fullfile(snapDir, '*.mat'));
    end
    
    dataTypes = {'bool', 'uint8', 'uint16', 'int16', 'uint32', 'int32', 'single', 'double'};
    
    cmdInfo = HIP.Info();
    ignoreCmds = {'Help','Info','DeviceCount','DeviceStats'};
    chkCmds = ({cmdInfo.command}.');
    
    bValidCmd = cellfun(@(x)(~any(strcmpi(x,ignoreCmds))), chkCmds);
    chkCmds = chkCmds(bValidCmd);
    
    cmdMap = containers.Map(chkCmds,zeros(length(chkCmds),1));
    
    for numdims=2:5
        rect = load(fullfile(testDir,'Images',['test_image_' num2str(numdims) 'd_rect.mat']));
        noise = load(fullfile(testDir,'Images',['test_image_' num2str(numdims) 'd_noise.mat']));
        sumnr = load(fullfile(testDir,'Images',['test_image_' num2str(numdims) 'd_sum.mat']));
        
        for i=1:length(dataTypes)
            imRect = convertIm(rect.im, dataTypes{i});
            imNoise = convertIm(noise.im, dataTypes{i});
            imSum = convertIm(sumnr.im, dataTypes{i});
            
            runAllCommands(snapDir, imRect, imNoise, imSum, numdims, dataTypes{i}, cmdMap);
        end
    end
    
    cmds = keys(cmdMap);
    for i=1:length(cmds)
        if ( cmdMap(cmds{i}) < 1 )
            warning(['Command HIP.' cmds{i} '() not run. Snapshot will not include output from this command.']);
        end
    end
    
    toc();
end

function imOut = convertIm(imIn, dataType)
    switch dataType
        case 'bool'
            imOut = (imIn >= 0.5);
        case 'uint8'
            imOut = uint8((2^8-1)*imIn);
        case 'uint16'
            imOut = uint16((2^14-1)*imIn);
        case 'int16'
            imOut = int16((2^14-1)*(imIn-0.5));
        case 'uint32'
            imOut = uint32((2^20-1)*imIn);
        case 'int32'
            imOut = int32((2^20-1)*(imIn-0.5));
        case 'single'
            imOut = single(imIn);
        case 'double'
            imOut = imIn;
    end
end

function runAllCommands(outDir, imRect, imNoise, imSum, numdims, dataType, cmdMap)
    %% For now mostly run on summed image
    
    pxSize = [1,1,3];
    numsdims = min(3, numdims);

    fullkerndims = [5,5,3];
    kerndims = fullkerndims(1:numsdims);
    
    kernel = ones(kerndims, 'single');
    sigmas = [10 ./ pxSize(1:numsdims) zeros(1,3-numsdims)];
    highpassSigmas = 2*sigmas;

    imOut = HIP.Closure(imSum, kernel, [],[]);
    saveCmdOutput(outDir, imOut, 'Closure', numdims, dataType, cmdMap);

    imOut = HIP.ElementWiseDifference(imSum, imNoise, []);
    saveCmdOutput(outDir, imOut, 'ElementWiseDifference', numdims, dataType, cmdMap);

    imOut = HIP.EntropyFilter(imSum, kernel, []);
    saveCmdOutput(outDir, imOut, 'EntropyFilter', numdims, dataType, cmdMap);

    imOut = HIP.Gaussian(imSum, sigmas, [],[]);
    saveCmdOutput(outDir, imOut, 'Gaussian', numdims, dataType, cmdMap);

    [imMin,imMax] = HIP.GetMinMax(imSum, []);
    imOut = [imMin,imMax];
    saveCmdOutput(outDir, imOut, 'GetMinMax', numdims, dataType, cmdMap);

%     imOut = HIP.HighPassFilter(imSum, highpassSigmas, []);
%     saveCmdOutput(outDir, imOut, 'HighPassFilter', numdims, dataType, cmdMap);

    imOut = HIP.LoG(imSum, sigmas, []);
    saveCmdOutput(outDir, imOut, 'LoG', numdims, dataType, cmdMap);

    imOut = HIP.MaxFilter(imSum, kernel, [], []);
    saveCmdOutput(outDir, imOut, 'MaxFilter', numdims, dataType, cmdMap);

    imOut = HIP.MeanFilter(imSum, kernel, [], []);
    saveCmdOutput(outDir, imOut, 'MeanFilter', numdims, dataType, cmdMap);

    imOut = HIP.MedianFilter(imSum, kernel, [], []);
    saveCmdOutput(outDir, imOut, 'MedianFilter', numdims, dataType, cmdMap);

    imOut = HIP.MinFilter(imSum, kernel, [], []);
    saveCmdOutput(outDir, imOut, 'MinFilter', numdims, dataType, cmdMap);

    imOut = HIP.MultiplySum(imSum, kernel, [], []);
    saveCmdOutput(outDir, imOut, 'MultiplySum', numdims, dataType, cmdMap);

    imOut = HIP.Opener(imSum, kernel, [], []);
    saveCmdOutput(outDir, imOut, 'Opener', numdims, dataType, cmdMap);

    imOut = HIP.StdFilter(imSum, kernel, [], []);
    saveCmdOutput(outDir, imOut, 'StdFilter', numdims, dataType, cmdMap);

    imOut = HIP.Sum(imSum, []);
    saveCmdOutput(outDir, imOut, 'Sum', numdims, dataType, cmdMap);

    imOut = HIP.VarFilter(imSum, kernel, [], []);
    saveCmdOutput(outDir, imOut, 'VarFilter', numdims, dataType, cmdMap);

    imOut = HIP.WienerFilter(imSum, kernel, [], []);
    saveCmdOutput(outDir, imOut, 'WienerFilter', numdims, dataType, cmdMap);
end

function saveCmdOutput(outDir, imOut, cmdStr, numdims, dataType, cmdMap)
    save(fullfile(outDir, [cmdStr '_' num2str(numdims) 'd_' dataType '.mat']), 'imOut');
    cmdMap(cmdStr) = cmdMap(cmdStr) + 1;
end

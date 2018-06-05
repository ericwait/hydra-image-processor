function times = GaussianFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials,numDevices)
    % times has size of image, cuda time, matlab time, cuda times faster,
    %   matlab over cuda
    % third dimension is type
    numItters = length(sizeItter)*length(typeItter)*numTrials;
    
    times = ones(size(sizes_rc,1),6,length(typeItter))*inf;
    prgs = Utils.CmdlnProgress(numItters,true,'GaussianFilter');
    k = 0;

    m = memory;
    cpu_memAvail = m.MemAvailableAllArrays/4;
    m = ImProc.Cuda.DeviceStats;
    gpu_memAvail = max(m.totalMem);
    
    for i = sizeItter
        for ty = typeItter
            imSizePerFrame = prod(sizes_rc(i,1:3))*(2^ty/2);
            imSize = prod(sizes_rc(i,:))*(2^ty/2);
            if (imSize<=cpu_memAvail && imSizePerFrame<=gpu_memAvail)
                im = ones(sizes_rc(i,:),types{ty});
                times(i,1,ty) = numel(im);
                
                ts = zeros(numTrials,3);
                for j=1:numTrials
                    [ts(j,1),ts(j,2),~,ts(j,3)] = Performance.GaussianFilter(im,[35,35,15],numDevices);
                    k = k +1;
                    prgs.PrintProgress(k);
                end
                times(i,2:4,ty) = mean(ts,1);
            end
        end
    end
    prgs.ClearProgress(true);
    times(:,5,:) = times(:,3,:)./times(:,2,:);
    times(:,6,:) = times(:,3,:)./times(:,4,:);

    Performance.PlotResults(times,'Gaussian Filter [35,35,15]',numDevices);
end
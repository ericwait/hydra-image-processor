function times = MaxFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials,numDevices)
    % times has size of image, cuda time, matlab time, cuda times faster,
    %   matlab over cuda
    % third dimension is type
    numItters = length(sizeItter)*length(typeItter)*numTrials;
    
    times = zeros(length(sizeItter),6,length(typeItter));
    prgs = Utils.CmdlnProgress(numItters,true,'MaxFilter');
    k = 0;
    kernel = ones(5,5,3);
    
    for i = sizeItter
        szImage = rand(2^sizes_rc(i),2^sizes_rc(i),2^(sizes_rc(i)-4),2);
        for ty = typeItter
            im = ImUtils.ConvertType(szImage,types{ty});
            times(i,1,ty) = numel(im);
            
            ts = zeros(numTrials,3);
            for j=1:numTrials
                [ts(j,1),ts(j,2),~,ts(j,3)] = Performance.MaxFilter(im,kernel,numDevices);
                k = k +1;
                prgs.PrintProgress(k);
            end
            times(i,2:4,ty) = mean(ts,1);
        end
    end
    prgs.ClearProgress(true);
    times(:,5,:) = times(:,3,:)./times(:,2,:);
    times(:,6,:) = times(:,3,:)./times(:,4,:);

    Performance.PlotResults(times,'Max Filter [5,5,3]',numDevices);
end
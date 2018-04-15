function times = MeanFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials,numDevices)
    % times has size of image, cuda time, matlab time, cuda times faster,
    %   matlab over cuda
    % third dimension is type
    numItters = length(sizeItter)*length(typeItter);
    
    times = zeros(length(sizeItter),7,length(typeItter));
    prgs = Utils.CmdlnProgress(numItters,true,'MeanFilter');
    j = 0;
    kernel = ones(5,5,3,'single');
    
    for i = sizeItter
        szImage = rand(2^sizes_rc(i),2^sizes_rc(i),2^(sizes_rc(i)-4),2,3);
        for ty = typeItter
            im = ImUtils.ConvertType(szImage,types{ty},true);
            times(i,1,ty) = numel(im);

            ts = zeros(numTrials,4);
            for j=1:numTrials
                [ts(j,1),ts(j,2),~,ts(j,3:4)] = Performance.MeanFilter(im,kernel,numDevices);
            end
            times(i,2:5,ty) = mean(ts,1);
            j = j +1;
            prgs.PrintProgress(j);
        end
    end
    prgs.ClearProgress(true);
    times(:,5,:) = times(:,3,:)./times(:,2,:);
    times(:,6,:) = times(:,3,:)./times(:,4,:);

    Performance.PlotResults(times,'MeanFilter [5,5,3]',numDevices);
end
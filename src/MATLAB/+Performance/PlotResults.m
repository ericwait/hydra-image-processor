function PlotResults(times,kernelName,numDevices)
    types = {'uint8';'uint16';'single';'double'};
    
% times has size of image, cuda time, matlab time, cuda times faster,
%   matlab over cuda
% third dimension is type
    
    f = figure;
    multiDevs = false;
    if (times(end,4,end)>1)
        multiDevs = true;
    end
    
    subIdx = [1,2,5,6];
    for i=1:size(types,1)
        subplot(2,4,subIdx(i));
        loglog(times(:,1,i),times(:,2,i),'-s','markersize',16);
        hold on
        loglog(times(:,1,i),times(:,3,i),'-s','markersize',16);
        if (numDevices>1)
            loglog(times(:,1,i),times(:,4,i),'-*','markersize',16);
            legend('Cuda','Matlab',sprintf('Cuda %d devices',numDevices),'Location','northwest');
        else
            legend('Cuda','Matlab','Location','northwest');
        end
        hold off
        xlabel('Number of Voxels');
        ylabel('Execution Time (sec)');
        title(types{i});
    end

    subplot(2,4,[3,4,7,8]);
    for i=1:size(types,1)
        loglog(times(:,1,i),times(:,5,i),'-s','markersize',16);
        hold on
    end
    
    if (numDevices>1)
        for i=1:size(types,1)
            loglog(times(:,1,i),times(:,6,i),':*','markersize',16);
        end
        typesDev = cellfun(@(x)([x,' multidevice']),types,'UniformOutput',false);
        types = vertcat(types,typesDev);
    end
    
    legend(types,'Location','northwest');

    title(kernelName)
    xlabel('Number of Voxels');
    ylabel('Cuda is this many times faster');
    
    f.Units = 'normalized';
    f.Position = [0,0,1,1];
    drawnow
end

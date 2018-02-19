function PlotResults(times,kernelName)
    types = {'uint8';'uint16';'single';'double'};
    
% times has size of image, cuda time, matlab time, cuda times faster,
%   matlab over cuda
% third dimension is type
    
    f = figure;
    subIdx = [1,2,5,6];
    for i=1:size(types,1)
        subplot(2,4,subIdx(i));
        loglog(times(:,1,i),times(:,2,i),'-s');
        %plot(times(:,1,i),times(:,2,i),'-s');
        hold on
        loglog(times(:,1,i),times(:,3,i),'-s');
        %plot(times(:,1,i),times(:,3,i),'-s');
        hold off
        legend('Cuda','Matlab','Location','northwest');
        xlabel('Number of Voxels');
        ylabel('Execution Time (sec)');
        title(types{i});
    end

    subplot(2,4,[3,4,7,8]);
    for i=1:size(types,1)
        loglog(times(:,1,i),times(:,5,i),'-s');
        %plot(times(:,1,i),times(:,5,i),'-s');
        hold on
    end
    
    legend(types,'Location','northwest');

    title(kernelName)
    xlabel('Number of Voxels');
    ylabel('Cuda is this many times faster');
    
    f.Units = 'normalized';
    f.Position = [0,0,1,1];
    drawnow
end

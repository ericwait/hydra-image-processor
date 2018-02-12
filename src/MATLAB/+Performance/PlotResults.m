function PlotResults(times,kernelName)
    figure
    subplot(1,2,1);
    plot(times(:,1),times(:,2));
    hold on
    plot(times(:,1),times(:,3));
    hold off
    legend('Cuda','Matlab','Location','northwest');
    xlabel('Number of Voxels');
    ylabel('Execution Time (sec)');

    subplot(1,2,2)
    plot(times(:,1),times(:,5));
    title(kernelName)
    xlabel('Number of Voxels');
    ylabel('Cuda is this many times faster');
end

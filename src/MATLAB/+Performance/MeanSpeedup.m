function mSpeedup = MeanSpeedup(times,device)
    if (~exist('device','var') || isempty(device))
        device = 1;
    end
    
    mSpeedup = times(:,5+device-1,:);
    mSpeedup = mSpeedup(~isnan(mSpeedup));
    mSpeedup = mean(mSpeedup(1:end-1));
end

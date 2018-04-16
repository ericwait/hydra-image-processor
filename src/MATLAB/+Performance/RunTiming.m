ImProc.BuildScript;

%%
numTrials = 4;

m = memory;
numDevices = ImProc.Cuda.DeviceCount();
sizes_rc = 5:14;
sizesMask = sizes_rc <= (round(log2((m.MemAvailableAllArrays/4)^(1/3)))-1);
sizes_rc = sizes_rc(sizesMask);
if (~any(sizes_rc))
    sizes_rc = 5;
end
sizeItter = length(sizes_rc):-1:1;
types = {'uint8';'uint16';'single';'double'};
typeItter = size(types,1):-1:1;

%% Max Filter
maxTimes = Performance.MaxFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials,numDevices);

%% Closure 
closeTimes = Performance.ClosureGraph(sizes_rc,sizeItter,types,typeItter,numTrials,numDevices);

%% Mean Filter
meanTimes = Performance.MeanFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials,numDevices);

%% Median Filter
medTimes = Performance.MedianFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials,numDevices);

%% Std Filter
stdTimes = Performance.StdFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials,numDevices);

%% GaussianFilter
sizeItterSm = sizeItter(2:end);
if (isempty(sizeItterSm))
    sizeItterSm = 1;
end
gaussTimes = Performance.GaussianFilterGraph(sizes_rc,sizeItterSm,types,typeItter,numTrials,numDevices);

%% EntropyFilter
%entropyTimes = Performance.EntropyFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Contrast Enhancement
sizeItterSm = sizeItter(2:end);
if (isempty(sizeItterSm))
    sizeItterSm = 1;
end
hpTimes = Performance.HighPassFilterGraph(sizes_rc,sizeItterSm,types,typeItter,numTrials,numDevices);

%% Save out results
temp = what('ImProc');
ImProcPath = temp.path;
compName = getenv('computername');

save(fullfile(ImProcPath,[compName,'.mat']),'maxTimes','closeTimes','meanTimes','medTimes','stdTimes','medTimes','gaussTimes','hpTimes');
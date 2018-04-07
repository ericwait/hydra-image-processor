ImProc.BuildScript;

%%
sizes_rc = [5,6,7,8,9,10,11];%,12];%,13];
sizeItter = length(sizes_rc):-1:1;
numTrials = 3;
types = {'uint8';'uint16';'single';'double'};
typeItter = size(types,1):-1:1;

%% Max Filter
maxTimes = Performance.MaxFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Closure 
closeTimes = Performance.ClosureGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Mean Filter
meanTimes = Performance.MeanFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Median Filter
medTimes = Performance.MedianFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Std Filter
stdTimes = Performance.StdFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% GaussianFilter
gaussTimes = Performance.GaussianFilterGraph(sizes_rc,sizeItter(1:end-1),types,typeItter,numTrials);

%% EntropyFilter
%entropyTimes = Performance.EntropyFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Contrast Enhancement
hpTimes = Performance.HighPassFilterGraph(sizes_rc,sizeItter(1:end-1),types,typeItter,numTrials);

%% Save out results
temp = what('ImProc');
ImProcPath = temp.path;
compName = getenv('computername');

save(fullfile(ImProcPath,[compName,'.mat']),'maxTimes','closeTimes','meanTimes','medTimes','stdTimes','medTimes','gaussTimes','hpTimes');
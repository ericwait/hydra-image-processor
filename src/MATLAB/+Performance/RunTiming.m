sizes_rc = [5,6,7,8,9,10,11,12];%,13];
sizeItter = length(sizes_rc):-1:1;
numTrials = 2;
types = {'uint8';'uint16';'single';'double'};
typeItter = size(types,1):-1:1;
%itter = 1:length(sizes_rc);

%% Add Constant
addTimes = Performance.AddConstantGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Add Image With
addWtimes = Performance.AddImageWithGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Contrast Enhancement
conEnTimes = Performance.ContrastEnhancementGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% EntropyFilter
entropyTimes = Performance.EntropyFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% GaussianFilter
gaussTimes = Performance.GaussianFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Median Filter
medTimes = Performance.MedianFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Std Filter
stdTimes = Performance.StdFilterGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Sum Array
sumTimes = Performance.SumArrayGraph(sizes_rc,sizeItter,types,typeItter,numTrials);

%% Save out results
temp = what('ImProc');
ImProcPath = temp.path;
compName = getenv('computername');

save(fullfile(ImProcPath,[compName,'.mat']),'addTimes','addWtimes','conEnTimes','entropyTimes','gaussTimes','medTimes','stdTimes','sumTimes');
ImProc.BuildScript;

%%
numTrials = 2;

numDevices = ImProc.Cuda.DeviceCount();
sizes_rc = [...
    0512,0512,1,1,1; % single small images
    1024,1024,1,1,1; % single medium image 
    2048,2048,1,1,1; % single large image
    1024,1024,150,1,1; % 3D medium image
    1024,1024,150,2,1; % 4D medium image
    1024,1024,150,3,2; % multispectrial timelapse
    2048,2048,800,2,2; % MultiView timelapse
    10000,10000,75,6,1; % huge 4D image
    ];

sizeItter = size(sizes_rc,1):-1:1;
%sizeItter = 1:size(sizes_rc,1);
types = {'uint8';'uint16';'single';'double'};
typeItter = size(types,1):-1:1;
%typeItter = 1:size(types,1);

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
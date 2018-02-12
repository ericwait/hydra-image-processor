sizes_rc = [5,6,7,8];%,9,10];%,11,12,13];
itter = length(sizes_rc):-1:1;
numItter = 4;
%itter = 1:length(sizes_rc);

%% Add Constant
addTimes = zeros(0,5);
prgs = Utils.CmdlnProgress(length(sizes_rc),true,'AddConstant');
for i = itter
    im = rand(2^sizes_rc(i),2^sizes_rc(i),'single');
    im = ImUtils.ConvertType(im,'uint16',true);
    im = repmat(im,[1,1,2^(sizes_rc(i)-4)]);
    addTimes(i,1) = numel(im);
    
    ts = zeros(numItter,2);
    for j=1:numItter
        [ts(j,1),ts(j,2)] = Performance.AddConstant(im,2);
    end
    addTimes(i,2:3) = mean(ts,1);
    prgs.PrintProgress(length(sizes_rc)-i +1);
end
prgs.ClearProgress(true);
clear im
addTimes(:,4) = addTimes(:,2)./addTimes(:,3)*100;
addTimes(:,5) = addTimes(:,3)./addTimes(:,2);

Performance.PlotResults(addTimes(2:end,:),'AddConstant');

%% Add Image With
addImWTimes = zeros(0,5);
prgs = Utils.CmdlnProgress(length(sizes_rc),true,'AddImageWith');
for i = itter
    im = rand(2^sizes_rc(i),2^sizes_rc(i),'single');
    im = ImUtils.ConvertType(im,'uint16',true);
    im = repmat(im,[1,1,2^(sizes_rc(i)-4)]);
    addImWTimes(i,1) = numel(im);
    
	ts = zeros(numItter,2);
	for j=1:numItter
		[ts(j,1),ts(j,2)] = Performance.AddImageWith(im);
	end
    addImWTimes(i,2:3) = mean(ts,1);

    prgs.PrintProgress(length(sizes_rc)-i +1);
end
prgs.ClearProgress(true);
clear im
addImWTimes(:,4) = addImWTimes(:,2)./addImWTimes(:,3)*100;
addImWTimes(:,5) = addImWTimes(:,3)./addImWTimes(:,2);

Performance.PlotResults(addImWTimes(2:end,:),'AddImageWith');

%% Contrast Enhancement
conEnTimes = zeros(0,5);
prgs = Utils.CmdlnProgress(length(sizes_rc),true,'ContrastEnhancement');
for i = itter
    im = rand(2^sizes_rc(i),2^sizes_rc(i),'single');
    im = ImUtils.ConvertType(im,'uint16',true);
    im = repmat(im,[1,1,2^(sizes_rc(i)-4)]);
    conEnTimes(i,1) = numel(im);
    
	ts = zeros(numItter,2);
	for j=1:numItter
		[ts(j,1),ts(j,2)] = Performance.ContrastEnhancement(im,[35,35,15],[3,3,3]);
	end
    conEnTimes(i,2:3) = mean(ts,1);

    prgs.PrintProgress(length(sizes_rc)-i +1);
end
prgs.ClearProgress(true);
clear im
conEnTimes(:,4) = conEnTimes(:,2)./conEnTimes(:,3)*100;
conEnTimes(:,5) = conEnTimes(:,3)./conEnTimes(:,2);

Performance.PlotResults(conEnTimes(2:end,:),'ContrastEnhancement');

%% EntropyFilter
entropyTimes = zeros(0,5);
prgs = Utils.CmdlnProgress(length(sizes_rc),true,'EntropyFilter');
for i = itter
    im = rand(2^sizes_rc(i),2^sizes_rc(i),'single');
    im = ImUtils.ConvertType(im,'uint16',true);
    im = repmat(im,[1,1,2^(sizes_rc(i)-4)]);
    entropyTimes(i,1) = numel(im);
    
	ts = zeros(numItter,2);
	for j=1:numItter
		[ts(j,1),ts(j,2)] = Performance.EntropyFilter(im,[5,5,3]);
	end
    entropyTimes(i,2:3) = mean(ts,1);

    prgs.PrintProgress(length(sizes_rc)-i +1);
end
prgs.ClearProgress(true);
clear im
entropyTimes(:,4) = entropyTimes(:,2)./entropyTimes(:,3)*100;
entropyTimes(:,5) = entropyTimes(:,3)./entropyTimes(:,2);

Performance.PlotResults(entropyTimes(2:end,:),'EntropyFilter');

%% GaussianFilter
gaussTimes = zeros(0,5);
prgs = Utils.CmdlnProgress(length(sizes_rc),true,'GaussianFilter');
for i = itter
    im = rand(2^sizes_rc(i),2^sizes_rc(i),'single');
    im = ImUtils.ConvertType(im,'uint16',true);
    im = repmat(im,[1,1,2^(sizes_rc(i)-4)]);
    gaussTimes(i,1) = numel(im);
    
	ts = zeros(numItter,2);
	for j=1:numItter
		[ts(j,1),ts(j,2)] = Performance.GaussianFilter(im,[35,35,15]);
	end
    gaussTimes(i,2:3) = mean(ts,1);

    prgs.PrintProgress(length(sizes_rc)-i +1);
end
prgs.ClearProgress(true);
clear im
gaussTimes(:,4) = gaussTimes(:,2)./gaussTimes(:,3)*100;
gaussTimes(:,5) = gaussTimes(:,3)./gaussTimes(:,2);

Performance.PlotResults(gaussTimes(2:end,:),'GaussianFilter');

%% Median Filter
medTimes = zeros(0,5);
prgs = Utils.CmdlnProgress(length(sizes_rc),true,'MedianFilter');
for i = itter
    im = rand(2^sizes_rc(i),2^sizes_rc(i),'single');
    im = ImUtils.ConvertType(im,'uint16',true);
    im = repmat(im,[1,1,2^(sizes_rc(i)-4)]);
    medTimes(i,1) = numel(im);
    
	ts = zeros(numItter,2);
	for j=1:numItter
		[ts(j,1),ts(j,2)] = Performance.MedianFilter(im,[5,5,3]);
	end
    medTimes(i,2:3) = mean(ts,1);

    prgs.PrintProgress(length(sizes_rc)-i +1);
end
prgs.ClearProgress(true);
clear im
medTimes(:,4) = medTimes(:,2)./medTimes(:,3)*100;
medTimes(:,5) = medTimes(:,3)./medTimes(:,2);

Performance.PlotResults(medTimes(2:end,:),'MedianFilter');

%% Std Filter
stdTimes = zeros(0,5);
prgs = Utils.CmdlnProgress(length(sizes_rc),true,'StdFilter');
for i = itter
    im = rand(2^sizes_rc(i),2^sizes_rc(i),'single');
    im = ImUtils.ConvertType(im,'uint16',true);
    im = repmat(im,[1,1,2^(sizes_rc(i)-4)]);
    stdTimes(i,1) = numel(im);
    
	ts = zeros(numItter,2);
	for j=1:numItter
		[ts(j,1),ts(j,2)] = Performance.StdFilter(im,[5,5,3]);
	end
    stdTimes(i,2:3) = mean(ts,1);

    prgs.PrintProgress(length(sizes_rc)-i +1);
end
prgs.ClearProgress(true);
clear im
stdTimes(:,4) = stdTimes(:,2)./stdTimes(:,3)*100;
stdTimes(:,5) = stdTimes(:,3)./stdTimes(:,2);

Performance.PlotResults(stdTimes(2:end,:),'StdFilter');

%% Sum Array
sumTimes = zeros(0,5);
prgs = Utils.CmdlnProgress(length(sizes_rc),true,'SumArray');
for i = itter
    im = rand(2^sizes_rc(i),2^sizes_rc(i),'single');
    im = ImUtils.ConvertType(im,'uint16',true);
    im = repmat(im,[1,1,2^(sizes_rc(i)-4)]);
    sumTimes(i,1) = numel(im);
    
	ts = zeros(numItter,2);
	for j=1:numItter
		[ts(j,1),ts(j,2)] = Performance.SumArray(im);
	end
    sumTimes(i,2:3) = mean(ts,1);

    prgs.PrintProgress(length(sizes_rc)-i +1);
end
prgs.ClearProgress(true);
clear im
sumTimes(:,4) = sumTimes(:,2)./sumTimes(:,3)*100;
sumTimes(:,5) = sumTimes(:,3)./sumTimes(:,2);

Performance.PlotResults(sumTimes(2:end,:),'SumArray');

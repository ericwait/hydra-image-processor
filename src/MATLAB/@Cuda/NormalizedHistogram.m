% NormalizedHistogram - histogram = NormalizedHistogram(imageIn,numBins,min,max,device) 
function histogram = NormalizedHistogram(imageIn,numBins,min,max,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [histogram] = Cuda.Mex('NormalizedHistogram',imageIn,numBins,min,max,device);

    delete(mutexfile);
end

% NormalizedCovariance - normalizedCovariance = NormalizedCovariance(imageIn1,imageIn2,device) 
function normalizedCovariance = NormalizedCovariance(imageIn1,imageIn2,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [normalizedCovariance] = Cuda.Mex('NormalizedCovariance',imageIn1,imageIn2,device);

    delete(mutexfile);
end

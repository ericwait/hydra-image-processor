% ThresholdFilter - imageOut = ThresholdFilter(imageIn,threshold,device) 
function imageOut = ThresholdFilter(imageIn,threshold,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('ThresholdFilter',imageIn,threshold,device);

    delete(mutexfile);
end

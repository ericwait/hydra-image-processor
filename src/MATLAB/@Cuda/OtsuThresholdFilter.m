% OtsuThresholdFilter - imageOut = OtsuThresholdFilter(imageIn,alpha,device) 
function imageOut = OtsuThresholdFilter(imageIn,alpha,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('OtsuThresholdFilter',imageIn,alpha,device);

    delete(mutexfile);
end

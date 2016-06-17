% Variance - variance = Variance(imageIn,device) This will return the variance of an image.
function variance = Variance(imageIn,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [variance] = Cuda.Mex('Variance',imageIn,device);

    delete(mutexfile);
end

% ReduceImage - imageOut = ReduceImage(imageIn,reductionFactor,method,device) 
function imageOut = ReduceImage(imageIn,reductionFactor,method,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('ReduceImage',imageIn,reductionFactor,method,device);

    delete(mutexfile);
end

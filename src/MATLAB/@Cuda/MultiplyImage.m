% MultiplyImage - imageOut = MultiplyImage(imageIn,multiplier,device) 
function imageOut = MultiplyImage(imageIn,multiplier,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('MultiplyImage',imageIn,multiplier,device);

    delete(mutexfile);
end

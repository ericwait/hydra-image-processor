% LinearUnmixing - imageOut = LinearUnmixing(mixedImages,unmixMatrix,device) 
function imageOut = LinearUnmixing(mixedImages,unmixMatrix,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('LinearUnmixing',mixedImages,unmixMatrix,device);

    delete(mutexfile);
end

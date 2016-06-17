% MinMax - [min,max] = MinMax(imageIn,device) 
function [min,max] = MinMax(imageIn,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [min,max] = Cuda.Mex('MinMax',imageIn,device);

    delete(mutexfile);
end

% SumArray - sum = SumArray(imageIn,device) 
function sum = SumArray(imageIn,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [sum] = Cuda.Mex('SumArray',imageIn,device);

    delete(mutexfile);
end

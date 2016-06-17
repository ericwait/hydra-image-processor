% StdFilter - imageOut = StdFilter(imageIn,Neighborhood,device) 
function imageOut = StdFilter(imageIn,Neighborhood,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('StdFilter',imageIn,Neighborhood,device);

    delete(mutexfile);
end

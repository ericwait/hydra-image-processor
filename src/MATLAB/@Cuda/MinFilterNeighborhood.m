% MinFilterNeighborhood - imageOut = MinFilterNeighborhood(imageIn,Neighborhood,device) 
function imageOut = MinFilterNeighborhood(imageIn,Neighborhood,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('MinFilterNeighborhood',imageIn,Neighborhood,device);

    delete(mutexfile);
end

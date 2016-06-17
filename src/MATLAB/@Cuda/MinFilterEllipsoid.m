% MinFilterEllipsoid - imageOut = MinFilterEllipsoid(imageIn,radius,device) 
function imageOut = MinFilterEllipsoid(imageIn,radius,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('MinFilterEllipsoid',imageIn,radius,device);

    delete(mutexfile);
end

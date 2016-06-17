% RegionGrowing - maskOut = RegionGrowing(imageIn,kernel,mask,threshold,allowConnections,device) 
function maskOut = RegionGrowing(imageIn,kernel,mask,threshold,allowConnections,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [maskOut] = Cuda.Mex('RegionGrowing',imageIn,kernel,mask,threshold,allowConnections,device);

    delete(mutexfile);
end

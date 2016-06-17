% MorphologicalClosure - imageOut = MorphologicalClosure(imageIn,kernel,device) 
function imageOut = MorphologicalClosure(imageIn,kernel,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('MorphologicalClosure',imageIn,kernel,device);

    delete(mutexfile);
end
